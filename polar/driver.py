"""BLE connection and SDK-mode streaming for Polar Sense devices."""

from __future__ import annotations

import bleak
import asyncio
from enum import IntEnum
from typing import Callable
import numpy as np
from bitarray import bitarray
from bitarray.util import ba2int
import time
import datetime
from pathlib import Path

class Driver:
    """BLE connection and SDK-mode streaming for Polar Sense devices."""

    class Device:
        """Represents a single connected Polar Sense device.

        Handles the BLE connection lifecycle, enabling SDK streaming mode,
        negotiating measurement settings, and decoding incoming delta
        frames. Decoded samples are forwarded to ``stream_callback``.
        """

        class UUID:
            """BLE characteristic and service UUIDs exposed by the device."""

            #UUIDs for device properties
            MODEL_NBR         = '00002a24-0000-1000-8000-00805f9b34fb'
            MANUFACTURER_NAME = '00002a29-0000-1000-8000-00805f9b34fb'
            BATTERY_LEVEL     = '00002a19-0000-1000-8000-00805f9b34fb'

            # UUIDs for characteristics
            SERVICE = 'FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8'
            CONTROL = 'FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8'
            DATA    = 'FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8'

        class MeasurementType(IntEnum):
            """Measurement types supported by the Polar Verity Sense."""

            # Polar Verity Sense supports PPG, ACC, GYR
            PPG = 1 # a.u.
            ACC = 2 # Force per unit mass [g]
            GYR = 5 # degrees per second [dps]

        class Mode(IntEnum):
            """Special measurement-type value used to select a streaming mode."""

            # Special measurement type for SDK mode
            SDK = 9

        class OpCode(IntEnum):
            """Control point operation codes."""

            # Op codes for different operations
            GET_MEASUREMENT_SETTINGS = 0x01
            START_STREAM             = 0x02
            STOP_STREAM              = 0x03

        class ErrorCode(IntEnum):
            """Status codes returned by the device after a control point operation."""

            # Error codes that can appear after starting/stopping streams
            SUCCESS                          = 0
            ERROR_INVALID_OP_CODE            = 1
            ERROR_INVALID_MEASUREMENT_TYPE   = 2
            ERROR_NOT_SUPPORTED              = 3
            ERROR_INVALID_LENGTH             = 4
            ERROR_INVALID_PARAMETER          = 5
            ERROR_ALREADY_IN_STATE           = 6
            ERROR_INVALID_RESOLUTION         = 7
            ERROR_INVALID_SAMPLE_RATE        = 8
            ERROR_INVALID_RANGE              = 9
            ERROR_INVALID_MTU                = 10
            ERROR_INVALID_NUMBER_OF_CHANNELS = 11
            ERROR_INVALID_STATE              = 12
            ERROR_DEVICE_IN_CHARGER          = 13

        class SettingType(IntEnum):
            """Setting types used when requesting or starting a measurement stream."""

            # Setting types for data acquisition
            SAMPLE_RATE = 0x00
            RESOLUTION  = 0x01
            RANGE       = 0x02
            CHANNELS    = 0x04

        class LogLevel(IntEnum):
            """Logging verbosity levels, mirroring the standard library's."""

            # Logging levels
            DEBUG    = 10
            INFO     = 20
            WARNING  = 30
            ERROR    = 40
            CRITICAL = 50

        def __init__(
                self,
                device: bleak.backends.device.BLEDevice,
                save_dir: Path,
                loglevel: LogLevel = LogLevel.INFO,
                stream_callback: Callable[['Driver.Device', 'Driver.Device.MeasurementType', np.ndarray, np.ndarray], None] | None = None,
                disconnect_callback: Callable[['Driver.Device'], None] | None = None
            ) -> None:
            """
            Args:
                device: The BLE device to connect to, as discovered by bleak.
                save_dir: Directory under which this device's log file is written.
                loglevel: Minimum level printed/saved; ERROR and above disconnect and raise.
                stream_callback: Called with each decoded frame once streaming has started.
                disconnect_callback: Called when the device disconnects, for any reason.
            """

            # Input parameters
            self.device              = device
            self.save_dir            = save_dir
            self.loglevel            = loglevel
            self.stream_callback     = stream_callback
            self.disconnect_callback = disconnect_callback

            # Device parameters
            self.device_name    = device.name[12:]
            self.device_address = device.address

        async def start(self) -> None:
            """Connect to the device and start streaming all supported measurement types."""

            await self.connect()
            await self.read_properties()
            await self.read_measurement_types()
            await self.enable_sdk_mode()
            await self.get_measurement_settings()
            await self.start_streams()

        def log(self, level: LogLevel, msg: str) -> None:
            """Print/persist a log message; escalate to disconnect on ERROR+ or a disconnect notice.

            Args:
                level: Severity of this message.
                msg: Message text.

            Raises:
                RuntimeError: If level is ERROR or higher, or the message indicates a disconnect.
            """

            msg_str = '{} - {} - {}: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), self.device_name, level.name, msg)

            if level >= self.loglevel:
                if level >= self.LogLevel.ERROR:
                    self.disconnect()
                    raise RuntimeError(msg_str)
                else:
                    print(msg_str)

            if level >= self.LogLevel.WARNING and ('disconnect' in msg or 'Not connected' in msg):
                self.disconnect()
                raise RuntimeError('Device disconnected')

            if not hasattr(self, 'log_buffer'):
                self.log_buffer = []

            self.log_buffer.append(msg_str)

            if hasattr(self, 'timestamp_start'):

                save_subdir = self.save_dir / '{}-{}'.format(datetime.datetime.fromtimestamp(self.timestamp_start).strftime('%Y%m%d%H%M%S'), self.device_name)
                save_file   = save_subdir / 'log.txt'

                if not save_subdir.exists():
                    save_subdir.mkdir()

                while len(self.log_buffer) > 0:
                    with open(save_file, 'a') as f:
                        f.write('{}\n'.format(self.log_buffer[0]))
                        self.log_buffer.pop(0)

        async def connect(self) -> None:
            """Connect to the device over BLE, retrying up to 10 times."""

            # Do 10 attempts to connect to device
            for attempt in range(1, 11):
                try:
                    self.log(self.LogLevel.DEBUG, 'Attempting to connect with device {}, address {}, attempt {}'.format(self.device_name, self.device_address, attempt))

                    # Create Bleak client with disconnect callback
                    self.client = bleak.BleakClient(
                        address_or_ble_device = self.device,
                        disconnected_callback = self.disconnect
                    )

                    # Connect to client
                    await self.client.connect()

                    # Make sure connection is good
                    while not self.client.is_connected:
                        await asyncio.sleep(0.5)

                    self.log(self.LogLevel.INFO, 'Connected to device {}'.format(self.device_address))
                    return

                except Exception as e:

                    try:
                        await self.client.disconnect()
                        del self.client
                    except Exception:
                        pass

                    self.log(self.LogLevel.WARNING, 'Connection attempt {} failed due to error: {}'.format(attempt, str(e)))

                    await asyncio.sleep(1)
                    continue

            self.log(self.LogLevel.CRITICAL, 'No more connection attempts after {} failed attempts'.format(attempt))

        async def read_properties(self) -> None:
            """Read model number, manufacturer and battery level, retrying up to 10 times."""

            for attempt in range(1, 11):
                try:
                    self.log(self.LogLevel.DEBUG, 'Attempting to read device properties, attempt {}'.format(attempt))

                    self.model_number  = (await self.client.read_gatt_char(self.UUID.MODEL_NBR)).decode()
                    self.manufacturer  = (await self.client.read_gatt_char(self.UUID.MANUFACTURER_NAME)).decode()
                    self.battery_level = (await self.client.read_gatt_char(self.UUID.BATTERY_LEVEL)).decode()

                    self.log(self.LogLevel.INFO, 'Read device properties - model_number: {}, manufacturer: {}, battery_level: {}'.format(self.model_number, self.manufacturer, self.battery_level))
                    return

                except Exception as e:

                    if hasattr(self, 'model_number'):
                        del self.model_number
                    if hasattr(self, 'manufacturer'):
                        del self.manufacturer
                    if hasattr(self, 'battery_level'):
                        del self.battery_level

                    self.log(self.LogLevel.WARNING, 'Attempt {} to read device properties failed due to error: {}'.format(attempt, str(e)))

                    await asyncio.sleep(5)
                    continue

            self.log(self.LogLevel.CRITICAL, 'No more connection attempts after {} failed attempts'.format(attempt))

        async def read_measurement_types(self) -> None:
            """Read which measurement types the device supports and require PPG/ACC/GYR."""

            for attempt in range(1, 11):
                try:
                    self.log(self.LogLevel.DEBUG, 'Attempting to read measurement types, attempt {}'.format(attempt))

                    # Read device characteristics and convert to bit string
                    att_read         = await self.client.read_gatt_char(self.UUID.CONTROL)
                    meastype_enabled = ''.join(format(byte, '08b') for byte in att_read[1:])

                    # Check if control point response is given
                    if att_read[0] != 0x0f:
                        raise RuntimeError('Expected control point response 0x0f after requesting measurement types, got '.format(hex(att_read[0])))

                    # Loop through bit string to detect which measurement types are supported
                    self.measurement_types = []
                    for meastype_int, enabled in enumerate(meastype_enabled):
                        if enabled == '1' and meastype_int in self.MeasurementType.__members__.values():
                            self.measurement_types.append(self.MeasurementType(meastype_int))

                    # Check if necessary measurement types are present
                    if self.MeasurementType.ACC not in self.measurement_types:
                        raise RuntimeError('Measurement type ACC is not supported')
                    if self.MeasurementType.GYR not in self.measurement_types:
                        raise RuntimeError('Measurement type GYR is not supported')
                    if self.MeasurementType.PPG not in self.measurement_types:
                        raise RuntimeError('Measurement type PPG is not supported')

                    self.log(self.LogLevel.INFO, 'Read measurement types: {}'.format(', '.join([x.name for x in self.measurement_types])))
                    return

                except Exception as e:

                    if hasattr(self, 'measurement_types'):
                        del self.measurement_types

                    self.log(self.LogLevel.WARNING, 'Attempt {} to read measurement types failed due to error: {}'.format(attempt, str(e)))

                    await asyncio.sleep(5)
                    continue

            self.log(self.LogLevel.CRITICAL, 'No more attempts to read measurement types after {} failed attempts'.format(attempt))

        async def enable_sdk_mode(self) -> None:
            """Request SDK streaming mode and wait for the device to confirm it, retrying up to 10 times."""

            for attempt in range(1, 11):
                try:
                    self.log(self.LogLevel.DEBUG, 'Attempting to enable SDK mode, attempt {}'.format(attempt))

                    await self.client.start_notify(self.UUID.CONTROL, self.enable_sdk_mode_callback)
                    self.enable_sdk_mode_flag = False

                    await self.client.write_gatt_char(self.UUID.CONTROL, bytearray([self.OpCode.START_STREAM, self.Mode.SDK]), response=True)

                    for attempt_1 in range(1, 11):
                        if not self.enable_sdk_mode_flag:
                            self.log(self.LogLevel.DEBUG, 'Device has not confirmed enabling SDK mode after wait attempt {}'.format(attempt_1))
                        else:
                            self.log(self.LogLevel.INFO, 'Enabled SDK mode after attempt {}'.format(attempt_1))
                            await self.client.stop_notify(self.UUID.CONTROL)
                            del self.enable_sdk_mode_flag
                            return
                        await asyncio.sleep(0.5)

                except Exception as e:

                    if hasattr(self, 'enable_sdk_mode_flag'):
                        del self.enable_sdk_mode_flag

                    try:
                        await self.client.stop_notify(self.UUID.CONTROL)
                    except Exception:
                        pass

                    try:
                        await self.client.write_gatt_char(self.UUID.CONTROL, bytearray([self.OpCode.STOP_STREAM, self.Mode.SDK]), response=True)
                    except Exception:
                        pass

                    self.log(self.LogLevel.WARNING, 'Attempt {} to enable SDK mode failed due to error: {}'.format(attempt, str(e)))

                    await asyncio.sleep(1)
                    continue

            self.log(self.LogLevel.CRITICAL, 'No more attempts to enable SDK mode after {} failed attempts'.format(attempt))

        def enable_sdk_mode_callback(self, _: object, data: bytearray) -> None:
            """BLE notification callback confirming that SDK mode was enabled.

            Args:
                _: The notifying GATT characteristic; unused.
                data: Raw control point response bytes.
            """

            try:

                # Check if control point response is given
                if data[0] != 0xf0:
                    raise RuntimeError('Expected control point response 0xF0 after SDK mode enable request, got {}'.format(hex(data[0])))

                # Check if start stream OpCode is given
                if data[1] != self.OpCode.START_STREAM:
                    raise RuntimeError('Expected control op code START_STREAM, got {}'.format(hex(data[1])))

                # Check mode identifier
                if data[2] != self.Mode.SDK:
                    raise RuntimeError('Expected mode identifier SDK, got {}'.format(hex(data[2])))

                # Check status
                if data[3] != self.ErrorCode.SUCCESS:
                    raise RuntimeError('Expected status SUCCESS, got error code {}'.format(self.ErrorCode(data[3]).name))

                # If all checks passed, set flag to true
                if hasattr(self, 'enable_sdk_mode_flag'):
                    self.enable_sdk_mode_flag = True

            except Exception as e:
                self.log(self.LogLevel.ERROR, 'Callback on SDK mode reported error: {}'.format(str(e)))

        async def get_measurement_settings(self) -> None:
            """Request and collect the available settings for each supported measurement type."""

            for attempt in range(1, 11):
                try:
                    self.log(self.LogLevel.DEBUG, 'Attempting to get measurement settings, attempt {}'.format(attempt))

                    await self.client.start_notify(self.UUID.CONTROL, self.get_measurement_settings_callback)

                    self.measurement_settings = {}

                    for measurement_type in self.measurement_types:
                        for attempt_1 in range(1, 11):
                            self.log(self.LogLevel.DEBUG, 'Attempting to get measurement settings for type {}, attempt {}'.format(measurement_type.name, attempt_1))

                            await self.client.write_gatt_char(self.UUID.CONTROL, bytearray([self.OpCode.GET_MEASUREMENT_SETTINGS, measurement_type]), response=True)

                            for attempt_2 in range(1, 11):
                                if measurement_type not in self.measurement_settings:
                                    self.log(self.LogLevel.DEBUG, 'Device has not provided measurement settings for type {} after wait attempt {}'.format(measurement_type.name, attempt_2))
                                else:
                                    self.log(self.LogLevel.INFO, 'Device has provided measurement settings for type {}'.format(measurement_type.name))
                                    break
                                await asyncio.sleep(0.5)

                            if measurement_type in self.measurement_settings:
                                break

                            await asyncio.sleep(0.5)
                        else:
                            raise RuntimeError('Device has not provided measurement settings for type {} after {} wait attempts'.format(measurement_type.name, attempt_1))

                    await self.client.stop_notify(self.UUID.CONTROL)

                    self.log(self.LogLevel.INFO, 'Device provided all measurement settings')
                    return

                except Exception as e:

                    if hasattr(self, 'measurement_settings'):
                        del self.measurement_settings

                    try:
                        await self.client.stop_notify(self.UUID.CONTROL)
                    except Exception:
                        pass

                    self.log(self.LogLevel.WARNING, 'Attempt {} to get measurement settings failed due to error: {}'.format(attempt, str(e)))

                    await asyncio.sleep(1)
                    continue

            self.log(self.LogLevel.CRITICAL, 'No more attempts to get measurement settings after {} failed attempts'.format(attempt))

        def get_measurement_settings_callback(self, _: object, data: bytearray) -> None:
            """BLE notification callback that decodes one measurement type's available settings.

            Args:
                _: The notifying GATT characteristic; unused.
                data: Raw control point response bytes.
            """

            try:
                self.log(self.LogLevel.DEBUG, 'Callback on measurement settings called')

                if not hasattr(self, 'measurement_settings'):
                    self.log(self.LogLevel.DEBUG, 'Callback on measurement settings has nothing to do')
                    return

                # Check if control point response is given
                if data[0] != 0xf0:
                    raise RuntimeError('Expected control point response 0xF0, got {}'.format(hex(data[0])))

                # Check OP code
                if data[1] != self.OpCode.GET_MEASUREMENT_SETTINGS:
                    raise RuntimeError('Expected op code GET_MEASUREMENT_SETTINGS, got {}'.format(hex(data[1])))

                # Check measurement type
                if data[2] not in self.measurement_types:
                    raise RuntimeError('Measurement type {} unknown'.format(hex(data[2])))

                # Check status
                if data[4] != self.ErrorCode.SUCCESS:
                    raise RuntimeError('Expected status SUCCESS, got error code {}'.format(self.ErrorCode(data[4]).name))

                # Fetch settings
                measurement_type = self.MeasurementType(data[2])
                temp_settings    = {}

                offset = 5
                while offset < len(data):

                    setting_type = self.SettingType(data[offset])
                    array_length = int(data[offset+1])
                    settings_raw  = data[offset+2:offset+2+2*array_length]

                    settings = []
                    for i in range(int(np.ceil(len(settings_raw)/2))):
                        settings.append(int.from_bytes(settings_raw[2*i:2*i+2], byteorder='little', signed=False))

                    temp_settings[setting_type] = settings

                    offset += 2 + 2 * array_length

                self.measurement_settings[measurement_type] = temp_settings

            except Exception as e:
                self.log(self.LogLevel.ERROR, 'Callback on measurement settings reported error: {}'.format(str(e)))

        async def start_streams(self) -> None:
            """Start streaming every supported measurement type using its best available settings."""

            for attempt in range(1, 11):
                try:
                    self.log(self.LogLevel.DEBUG, 'Attempting to request start stream, attempt {}'.format(attempt))

                    # Choose best available stream settings
                    self.stream_settings = {}

                    for measurement_type in self.measurement_types:
                        self.stream_settings[measurement_type] = {}

                        for setting_type, settings in self.measurement_settings[measurement_type].items():
                            self.stream_settings[measurement_type][setting_type] = settings[-1]
                            self.log(self.LogLevel.DEBUG, 'Setting {} for type {} is set to {}'.format(setting_type.name, measurement_type.name, self.stream_settings[measurement_type][setting_type]))

                    # Prepare bluetooth chars to start streams
                    chars = {}

                    for measurement_type in self.measurement_types:

                        chars[measurement_type] = [self.OpCode.START_STREAM, measurement_type]

                        for setting_type, value in self.stream_settings[measurement_type].items():
                            # Settings are generally 2 bytes long, except for CHANNELS, which is 1 byte long
                            length  = 1 if setting_type == self.SettingType.CHANNELS else 2
                            setting = list(bytearray(int(value).to_bytes(length=length, byteorder='little', signed=False)))

                            chars[measurement_type] = chars[measurement_type] + [setting_type, 0x01] + setting

                        chars[measurement_type] = bytearray(chars[measurement_type])

                    # Prepare stream started flags
                    self.stream_started = {}
                    for measurement_type in self.measurement_types:
                        self.stream_started[measurement_type] = False

                    # Start notifiers
                    await self.client.start_notify(self.UUID.CONTROL, self.start_streams_callback)
                    await self.client.start_notify(self.UUID.DATA, self.receiver_callback)

                    # Start streams and await stream flags
                    for measurement_type in self.measurement_types:
                        for attempt_1 in range(1, 11):
                            self.log(self.LogLevel.DEBUG, 'Attempting to request start stream for type {}, attempt {}'.format(measurement_type.name, attempt_1))

                            await self.client.write_gatt_char(self.UUID.CONTROL, chars[measurement_type], response=True)

                            for attempt_2 in range(1, 11):
                                if self.stream_started[measurement_type] is False:
                                    self.log(self.LogLevel.DEBUG, 'Device has not started stream for type {} after wait attempt {}'.format(measurement_type.name, attempt_2))
                                else:
                                    self.log(self.LogLevel.INFO, 'Device has started stream for type {}'.format(measurement_type.name))
                                    break
                                await asyncio.sleep(0.5)

                            if self.stream_started[measurement_type] is True:
                                break

                            await asyncio.sleep(0.5)
                        else:
                            raise RuntimeError('Device has not started stream for type {} after {} wait attempts'.format(measurement_type.name, attempt_1))

                    # Stop CONTROL notifier
                    await self.client.stop_notify(self.UUID.CONTROL)

                    # Delete flags
                    del self.stream_started

                    self.log(self.LogLevel.INFO, 'All streams started')

                    return

                except Exception as e:

                    if hasattr(self, 'stream_started'):
                        del self.stream_started

                    try:
                        await self.client.stop_notify(self.UUID.CONTROL)
                    except Exception:
                        pass

                    try:
                        await self.client.stop_notify(self.UUID.DATA)
                    except Exception:
                        pass

                    for measurement_type in self.measurement_types:
                        await self.client.write_gatt_char(self.UUID.CONTROL, bytearray([self.OpCode.STOP_STREAM, measurement_type]), response=True)

                    self.log(self.LogLevel.WARNING, 'Attempt {} to request start stream failed due to error: {}'.format(attempt, str(e)))

                    await asyncio.sleep(1)
                    continue

            self.log(self.LogLevel.CRITICAL, 'No more attempts to request start stream after {} failed attempts'.format(attempt))

        def start_streams_callback(self, _: object, data: bytearray) -> None:
            """BLE notification callback confirming that a measurement type's stream has started.

            Args:
                _: The notifying GATT characteristic; unused.
                data: Raw control point response bytes.
            """

            try:
                self.log(self.LogLevel.DEBUG, 'Callback on start streams called')

                if not hasattr(self, 'stream_started'):
                    self.log(self.LogLevel.DEBUG, 'Callback on start streams has nothing to do')
                    return

                # Check if control point response is given
                if data[0] != 0xf0:
                    raise RuntimeError('Expected control point response 0xF0, got {}'.format(hex(data[0])))

                # Check OP code
                if data[1] != self.OpCode.START_STREAM:
                    raise RuntimeError('Expected op code START_STREAM, got {}'.format(hex(data[1])))

                # Check measurement type
                if data[2] not in self.measurement_types:
                    raise RuntimeError('Measurement type {} unknown'.format(hex(data[2])))

                # Check status
                if data[4] != self.ErrorCode.SUCCESS:
                    raise RuntimeError('Expected status SUCCESS, got error code {}'.format(self.ErrorCode(data[4]).name))

                # Fetch settings
                measurement_type = self.MeasurementType(data[2])

                if hasattr(self, 'stream_started'):
                    self.stream_started[measurement_type] = True

            except Exception as e:
                self.log(self.LogLevel.ERROR, 'Callback on start stream request reported error: {}'.format(str(e)))

        def receiver_callback(self, _: object, data: bytearray) -> None:
            """BLE notification callback that decodes a raw delta frame and forwards it to process_frame.

            Args:
                _: The notifying GATT characteristic; unused.
                data: Raw delta frame bytes (header plus reference and delta samples).
            """

            self.log(self.LogLevel.DEBUG, 'Processing delta frame')

            # Capture computer timestamp
            timestamp_comp_last = time.time_ns() / 1e9

            # Process header
            frame_header = data[0:10]

            measurement_type = self.MeasurementType(frame_header[0])

            # Polar timestamps are the number of nanoseconds since Jan 1, 2000, 00:00
            # Convert to seconds
            timestamp_last  = int.from_bytes(bytearray(frame_header[1:9]), byteorder="little", signed=False)
            timestamp_last /= 1e9

            frame_type = data[9]

            # To be sure, check frame type. Frame type should be delta frame
            if frame_type != 0x80 and frame_type != 0x81:
                self.log(self.LogLevel.CRITICAL, 'Unknown frametype {}'.format(frame_type))

            self.log(self.LogLevel.DEBUG, 'Header - type {}, timestamp: {}, frame_type: {}'.format(measurement_type.name, timestamp_last, frame_type))

            # Fetch resolution and channels from stream settings
            resolution = self.stream_settings[measurement_type][self.SettingType.RESOLUTION]
            channels   = self.stream_settings[measurement_type][self.SettingType.CHANNELS]

            self.log(self.LogLevel.DEBUG, 'Using settings - resolution: {}, channels: {}'.format(resolution, channels))

            # Process frame contents
            frame_contents = data[10:]
            frame_contents = bitarray([frame_contents[i//8] & 1 << i%8 != 0 for i in range(len(frame_contents) * 8)], endian='little')

            # Reference samples are always given as full bytes, while the resolution may not fit full bytes. Some bit padding may be applied before the samples
            reference_step = int(np.ceil(resolution/8)*8)

            reference_sample = frame_contents[0:reference_step*channels]
            reference_sample = [ba2int(reference_sample[i*reference_step:(i+1)*reference_step], signed=True) for i in range(channels)]

            self.log(self.LogLevel.DEBUG, 'Reference - reference_step: {}, reference_sample: {}'.format(reference_step, reference_sample))

            # Process delta packages
            offset        = channels*reference_step
            frame_samples = [reference_sample]

            while offset < len(frame_contents):

                self.log(self.LogLevel.DEBUG, 'Starting delta package at offset {}/{}'.format(offset, len(frame_contents)))

                # Fetch delta size and samples_count
                delta_size    = ba2int(frame_contents[offset:offset+8], signed=False)
                samples_count = ba2int(frame_contents[offset+8:offset+16], signed=False)

                self.log(self.LogLevel.DEBUG, 'Delta package settings - delta_size: {}. sample_count: {}'.format(delta_size, samples_count))

                for sample_num in range(samples_count):

                    samples = frame_contents[offset+16+sample_num*delta_size*channels:offset+16+(sample_num+1)*delta_size*channels]
                    samples = [ba2int(samples[i*delta_size:(i+1)*delta_size], signed=True) for i in range(channels)]

                    frame_samples.append(samples)

                # Go to next delta package, which starts at a full byte
                offset += 16 + samples_count*delta_size*channels
                offset = int(np.ceil(offset/8)*8)

            # To be sure: check whether we are really at the end of the delta frame
            if len(frame_contents) != offset:
                self.log(self.LogLevel.ERROR, 'len(frame_contents) != final offset')

            # When all samples have been collected, convert to numpy
            frame_samples = np.array(frame_samples)

            # Delta frames use differential encoding. Apply cumulative sum to get decoded values
            frame_samples = np.cumsum(frame_samples, axis=0)

            self.log(self.LogLevel.DEBUG, 'Completed processing delta frame')

            self.process_frame(measurement_type, timestamp_last, timestamp_comp_last, frame_samples)

        def process_frame(self, measurement_type: MeasurementType, timestamp_last: float, timestamp_comp_last: float, frame_samples: np.ndarray) -> None:
            """Turn a decoded frame into absolute timestamps and forward it to stream_callback.

            The device's first frame per measurement type is used only to establish the
            offset between the device's and computer's clocks; from the second frame
            onward, timestamps are computed and the frame is forwarded.

            Args:
                measurement_type: Which measurement type this frame belongs to.
                timestamp_last: Device-clock timestamp of the frame's last sample, in seconds.
                timestamp_comp_last: Computer-clock timestamp captured when the frame arrived.
                frame_samples: Decoded samples, shape (samples, channels).
            """

            if not hasattr(self, 'timestamps_last'):
                self.timestamps_last      = {}
                self.timestamps_comp_last = {}

            if not hasattr(self, 'timestamp_delta'):

                # Save timestamps of last received frames
                self.timestamps_last[measurement_type]      = timestamp_last
                self.timestamps_comp_last[measurement_type] = timestamp_comp_last

                # If timestamps for all measurement types are available, determine timestamp_delta and timestamp_start
                if all([mt in self.timestamps_last for mt in self.measurement_types]):

                    best_mt             = None
                    best_timestamp_last = np.inf
                    for mt in self.measurement_types:
                        if self.timestamps_last[mt] < best_timestamp_last:
                            best_mt             = mt
                            best_timestamp_last = self.timestamps_last[mt]

                    self.timestamp_start = self.timestamps_comp_last[best_mt]
                    self.timestamp_delta = self.timestamps_last[best_mt] - self.timestamps_comp_last[best_mt]

                    self.log(self.LogLevel.DEBUG, 'Set timestamp_start to {} and timestamp_delta to {}'.format(self.timestamp_start, self.timestamp_delta))

                    del self.timestamps_comp_last

            else:

                # Determine time vector for this frame
                fs_hat          = len(frame_samples) / (timestamp_last - self.timestamps_last[measurement_type])
                timestamp_first = timestamp_last - (len(frame_samples) - 1) / fs_hat
                timestamps      = np.linspace(timestamp_first, timestamp_last, len(frame_samples), dtype=np.float64)
                timestamps      = timestamps - self.timestamp_delta - self.timestamp_start

                # Set timestamp for the next frame
                self.timestamps_last[measurement_type] = timestamp_last

                # If stream callback is provided, pass data
                if self.stream_callback is not None:
                    self.stream_callback(self, measurement_type, timestamps, frame_samples)

        def disconnect(self, *args, **kwargs) -> None:
            """Invoke the disconnect callback, if any.

            Accepts and ignores arbitrary arguments so it can be used directly as
            bleak's ``disconnected_callback``.

            Args:
                *args: Ignored.
                **kwargs: Ignored.
            """

            if self.disconnect_callback is not None:
                self.disconnect_callback(self)

    class Manager:
        """Scans for Polar Sense devices in range and captures any newly found one."""

        def __init__(self, save_dir: Path, stream_callback: Callable[['Driver.Device', 'Driver.Device.MeasurementType', np.ndarray, np.ndarray], None] | None, loglevel: Driver.Device.LogLevel | None = None) -> None:
            """
            Args:
                save_dir: Directory passed through to each captured Driver.Device.
                stream_callback: Passed through to each captured Driver.Device.
                loglevel: Passed through to each captured Driver.Device. Defaults to INFO.
            """

            self.save_dir        = save_dir
            self.stream_callback = stream_callback
            self.loglevel        = loglevel if loglevel is not None else Driver.Device.LogLevel.INFO

            self.polar_devices = {}

        def check_device(self, device: bleak.backends.device.BLEDevice) -> bool:
            """Return whether a discovered BLE device is a not-yet-captured Polar Sense.

            Args:
                device: A device found by a BLE scan.

            Returns:
                True if the device is named "Polar Sense..." and isn't already tracked.
            """

            if device.name is None:
                return False
            elif 'Polar Sense' not in device.name:
                return False
            elif device.address in self.polar_devices:
                return False
            else:
                return True

        async def capture_device(self, device: bleak.backends.device.BLEDevice) -> None:
            """Connect to a device and start streaming from it, tracking it while connected.

            Args:
                device: A device found by a BLE scan, as accepted by check_device.
            """

            polar_device = Driver.Device(
                device              = device,
                save_dir            = self.save_dir,
                loglevel            = self.loglevel,
                stream_callback     = self.stream_callback,
                disconnect_callback = self.disconnect_callback
            )

            await polar_device.start()

            self.polar_devices[polar_device.device_address] = polar_device

            print('{} - MANAGER - INFO: Connected device {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), self.polar_devices[polar_device.device_address].device_name))

        def disconnect_callback(self, polar_device: Driver.Device) -> None:
            """Stop tracking a device once it has disconnected.

            Args:
                polar_device: The device that disconnected.
            """

            if polar_device.device_address in self.polar_devices:
                print('{} - MANAGER - INFO: Disconnected device {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), self.polar_devices[polar_device.device_address].device_name))
                del self.polar_devices[polar_device.device_address]

        async def run(self) -> None:
            """Continuously scan for and capture Polar Sense devices. Runs forever."""

            print('{} - MANAGER - INFO: Started Polar Sense Device Manager'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))

            while True:

                try:
                    async with bleak.BleakScanner() as scanner:
                        for device in await scanner.discover():
                            if self.check_device(device) is True:
                                await self.capture_device(device)

                except Exception as e:
                    print('{} - MANAGER - ERROR: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), str(e)))
