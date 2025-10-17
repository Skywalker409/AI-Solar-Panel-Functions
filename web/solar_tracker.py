#!/usr/bin/env python3                           # Shebang to run with Python 3 when executed directly

"""
solar_tracker.py                                 # Module name
Fully commented Raspberry Pi controller for      # Short description
dual-axis solar tracking using Adafruit TMC2209  # Hardware: TMC2209 drivers
(step/dir) + UART configuration and a Portescap  # We use STEP/DIR pulses and configure the driver via UART
NEMA-23 through 10:1 planetary and 5:1 bevel     # Your gear train is 10:1 * 5:1 = 50:1 overall
gearing.                                         # Clarifies mechanics

Includes:
- Homing (with switches or soft-home)
- Relative and absolute moves (degrees)
- Trapezoid accel/decel
- Backlash compensation
- Soft limits
- UART microstep/current setup for TMC2209
- Persistent state on disk
- A CLI to run from terminal (home, move-by, move-to, return, status, jog)
"""

# ===========================
# ---- Python std imports ----
# ===========================

import os                                         # For file path and atomic rename on state save
import time                                       # For delays and timing
import json                                       # For persisting last-known angles on disk
import math                                       # (Reserved for future trig ops if you add sun calc)
from typing import Optional, Tuple                # Type hints for clarity

# ============================
# ---- External libraries -----
# ============================

try:
    import RPi.GPIO as GPIO                       # Raspberry Pi GPIO library (bit-banged STEP pulses)
except ImportError as e:
    raise SystemExit(                             # Exit with an instructional message if not installed
        "Please install RPi.GPIO (sudo apt-get install -y python3-rpi.gpio)"
    ) from e

try:
    import serial                                 # PySerial for TMC2209 UART config
except ImportError as e:
    raise SystemExit(                             # Exit with install hint if missing
        "Please install pyserial (pip3 install pyserial)"
    ) from e

# ===========================
# ---- User configuration ----
# ===========================

# ---- Mechanics & kinematics ----
MOTOR_STEPS_PER_REV = 200                         # 1.8°/step → 200 full steps per motor revolution
MICROSTEPPING = 16                                # Fallback microstepping if UART is disabled/fails
GEAR_RATIO = 10 * 5                               # 10:1 planetary * 5:1 bevel = 50:1 overall reduction
STEPS_PER_PANEL_REV = MOTOR_STEPS_PER_REV * MICROSTEPPING * GEAR_RATIO  # Microsteps at the panel shaft
STEPS_PER_DEGREE = STEPS_PER_PANEL_REV / 360.0    # Microsteps needed to rotate panel by 1°

BACKLASH_DEG = 0.30                               # Backlash compensation (degrees at panel); tune by test

# ---- Motion profile (conservative defaults) ----
MAX_STEP_FREQ = 1500                              # Max step rate (Hz) in cruise; reduce if skipping steps
ACCEL_TIME   = 0.35                               # Seconds to ramp from MIN to MAX (linear ramp)
MIN_STEP_FREQ = 120                               # Starting/ending step rate (Hz) to ensure reliable steps

# ---- Software soft limits (panel degrees) ----
AZ_MIN_DEG, AZ_MAX_DEG = (-95.0, 95.0)            # Azimuth safe range; adjust to your build
EL_MIN_DEG, EL_MAX_DEG = (  0.0, 90.0)            # Elevation safe range; typically 0..90°

# ---- Homing configuration ----
USE_LIMIT_SWITCHES = True                         # True if you wired mechanical limit switches
LIMIT_ACTIVE_LOW = True                           # True if switches pull the pin to GND when pressed
AZ_HOME_TOWARD_POSITIVE = False                   # Which direction reaches the AZ "start" switch/edge
EL_HOME_TOWARD_POSITIVE = False                   # Which direction reaches the EL "start" switch/edge
SOFT_HOME_TRAVEL_DEG = 6.0                        # Distance to push into hard stop for soft-home (no switch)
HOME_BUMP_BACK_DEG   = 0.5                        # After hitting switch/stop, back off this much
HOME_OFFSET_DEG      = 0.0                        # Logical zero offset after homing (use if your zero isn’t at switch)

# ---- GPIO pin map (BCM numbering) ----
AZ_PINS = {                                       # Azimuth axis pins
    "STEP": 17,                                   # STEP pin
    "DIR":  27,                                   # DIR pin
    "EN":   22,                                   # Enable pin (active level depends on board; here LOW=enable)
    "MIN_SW":  5,                                 # Min/home switch (optional)
    "MAX_SW":  6,                                 # Max limit switch (optional)
}
EL_PINS = {                                       # Elevation axis pins
    "STEP": 23,                                   # STEP pin
    "DIR":  24,                                   # DIR pin
    "EN":   25,                                   # Enable pin
    "MIN_SW":  12,                                # Min/home switch
    "MAX_SW":  16,                                # Max limit switch
}

# ---- Persistent state (store on disk) ----
STATE_PATH = "/home/pi/solar_tracker_state.json"  # Path to store last known AZ/EL angles

# ---------------------------
# ---- UART configuration ----
# ---------------------------

ENABLE_UART = True                                # Master switch: True enables UART configuration at startup
UART_PORT   = "/dev/serial0"                      # Pi’s primary serial device after raspi-config
UART_BAUD   = 115200                              # TMC2209 default reliable baud rate

AZ_SLAVE_ADDR = 0                                 # TMC2209 slave address (set via board address pins) for AZ
EL_SLAVE_ADDR = 1                                 # TMC2209 slave address for EL

UART_MICROSTEPS = 256                             # Desired microsteps (allowed: 256,128,64,32,16,8,4,2,1)

IRUN       = 20                                   # Running current (0..31); higher = more torque/heat
IHOLD      = 8                                    # Holding current (0..31) for idle torque and heat
IHOLDDELAY = 6                                    # Delay before dropping to hold current (0..15)

ENABLE_STEALTHCHOP = True                         # Quiet mode at low speed (stealthChop on)
ENABLE_INTERPOLATE = True                         # microPlyer interpolation to internal 256 µsteps

# ======================================
# ---- TMC2209 UART helper (minimal) ----
# ======================================

def _crc8(data: bytes) -> int:                    # CRC-8 calculator for TMC UART frames
    """CRC-8 polynomial 0x07, init 0x00."""       # Docstring specifying CRC flavor
    crc = 0                                       # Start accumulator at 0
    for b in data:                                # Iterate over each byte
        crc ^= b                                  # XOR byte into CRC
        for _ in range(8):                        # Process 8 bits
            if crc & 0x80:                        # If MSB is set
                crc = ((crc << 1) ^ 0x07) & 0xFF  # Shift left and apply poly
            else:                                 # Else
                crc = (crc << 1) & 0xFF           # Just shift left
    return crc                                    # Return final 8-bit CRC

class TMC2209:                                    # Minimal class to write common TMC2209 registers
    # Register addresses we care about (from TMC2209 datasheet)
    REG_GCONF        = 0x00                        # Global configuration
    REG_IHOLD_IRUN   = 0x10                        # Motor currents
    REG_CHOPCONF     = 0x6C                        # Chopper + microstep resolution
    REG_PWMCONF      = 0x70                        # StealthChop config

    def __init__(self, port: str, addr: int, baud: int = UART_BAUD, timeout: float = 0.02):  # Constructor
        self.addr = addr & 0x03                    # Keep address in 0..3 (TMC2209 supports 4 addresses)
        self.ser  = serial.Serial(                 # Open serial port with given settings
            port=port, baudrate=baud, timeout=timeout
        )

    def _write_reg(self, reg: int, value: int) -> None:      # Low-level write helper
        """Write 32-bit value to a register over UART."""     # Docstring
        payload = bytes([                                     # Build the 7-byte payload (no CRC)
            0x05,                                             # Sync/Slave-Write marker
            self.addr,                                        # Device address (0..3)
            (reg & 0x7F) | 0x80,                              # Register address with write bit set (MSB=1)
            (value >> 24) & 0xFF,                             # Data byte 3 (MSB)
            (value >> 16) & 0xFF,                             # Data byte 2
            (value >>  8) & 0xFF,                             # Data byte 1
            (value >>  0) & 0xFF,                             # Data byte 0 (LSB)
        ])
        crc = _crc8(payload)                                  # Compute CRC over payload
        frame = payload + bytes([crc])                        # Append CRC to form full frame
        self.ser.write(frame)                                 # Send frame out the UART
        time.sleep(0.001)                                     # Small pause to let IC process

    def set_gconf(self, stealthchop: bool = True, interpolate: bool = True) -> None:  # Configure global mode
        """Set stealthChop (via en_spreadCycle=0) and interpolation (intpol=1)."""     # Docstring
        val = 0                                                # Start with 0 (clear all)
        if not stealthchop:                                    # If stealthChop disabled
            val |= (1 << 2)                                    # Set en_spreadCycle=1 → spreadCycle mode
        if interpolate:                                        # If interpolation desired
            val |= (1 << 9)                                    # Set intpol=1 → microPlyer on
        self._write_reg(self.REG_GCONF, val)                   # Push to chip

    def set_current(self, irun: int, ihold: int, iholddelay: int) -> None:   # Set motor currents
        """Set IRUN/IHOLD/IHOLDDELAY (value ranges clamped)."""              # Docstring
        irun  = max(0, min(31, irun))                          # Clamp IRUN to 0..31
        ihold = max(0, min(31, ihold))                         # Clamp IHOLD to 0..31
        ihdel = max(0, min(15, iholddelay))                    # Clamp IHOLDDELAY to 0..15
        val = (ihold & 0x1F)                                   # Bits 0..4 = IHOLD
        val |= ((irun & 0x1F) << 8)                            # Bits 8..12 = IRUN
        val |= ((ihdel & 0x0F) << 16)                          # Bits 16..19 = IHOLDDELAY
        self._write_reg(self.REG_IHOLD_IRUN, val)              # Write to driver

    # Mapping from external microstep setting to CHOPCONF MRES code:
    # MRES code: 0=256,1=128,2=64,3=32,4=16,5=8,6=4,7=2,8=1
    MRES_CODE = {256:0, 128:1, 64:2, 32:3, 16:4, 8:5, 4:6, 2:7, 1:8}  # Lookup table

    def set_microsteps(self, microsteps: int) -> None:         # Set MRES bits via CHOPCONF
        """Set external microstepping (1..256)."""             # Docstring
        code = self.MRES_CODE.get(microsteps)                   # Look up code
        if code is None:                                        # If invalid
            raise ValueError("microsteps must be one of {256,128,64,32,16,8,4,2,1}")  # Raise error
        TOFF = 3                                                # Reasonable default off time
        HEND = 0                                                # HEND default (spreadCycle param)
        HSTRT= 4                                                # HSTRT default (spreadCycle param)
        chop = (TOFF & 0x0F)                                    # Place TOFF in bits 0..3
        chop |= ((HEND & 0x0F)  << 4)                           # Place HEND   in bits 4..7
        chop |= ((HSTRT & 0x07) << 7)                           # Place HSTRT  in bits 7..9
        chop |= ((code & 0x0F)  << 24)                          # Place MRES   in bits 24..27
        self._write_reg(self.REG_CHOPCONF, chop)                # Write to CHOPCONF

    def set_pwmconf_default(self) -> None:                      # Configure PWM/stealthChop behavior
        """Enable autoscale/autograd; medium PWM frequency."""  # Docstring
        pwm_autoscale = 1                                       # Enable automatic amplitude scaling
        pwm_autograd  = 1                                       # Enable automatic gradient
        pwm_freq      = 1                                       # PWM frequency code (0..3), 1 is a good start
        val = 0                                                 # Start value
        val |= (pwm_freq & 0x03) << 16                          # Place pwm_freq into bits 16..17
        if pwm_autoscale: val |= (1 << 18)                      # Set autoscale bit
        if pwm_autograd:  val |= (1 << 19)                      # Set autograd bit
        self._write_reg(self.REG_PWMCONF, val)                  # Write PWMCONF
# End TMC2209 class                                              # End of UART driver section

# =================================
# ---- GPIO & motion primitives ----
# =================================

def _ensure_gpio_setup() -> None:                               # Initialize all GPIOs (outputs/inputs)
    GPIO.setmode(GPIO.BCM)                                      # Use Broadcom GPIO numbering
    GPIO.setwarnings(False)                                     # Suppress duplicate setup warnings
    for pins in (AZ_PINS, EL_PINS):                             # For both axes
        GPIO.setup(pins["STEP"], GPIO.OUT, initial=GPIO.LOW)    # STEP pin as output, start LOW
        GPIO.setup(pins["DIR"],  GPIO.OUT, initial=GPIO.LOW)    # DIR pin as output, default LOW
        GPIO.setup(pins["EN"],   GPIO.OUT, initial=GPIO.LOW)    # EN as output (assume LOW=enable)
        if USE_LIMIT_SWITCHES:                                  # If using limit switches
            for key in ("MIN_SW", "MAX_SW"):                    # For each switch pin
                if key in pins and pins[key] is not None:       # If the pin exists
                    if LIMIT_ACTIVE_LOW:                        # If active low logic
                        GPIO.setup(pins[key], GPIO.IN, pull_up_down=GPIO.PUD_UP)   # Use pull-up
                    else:                                       # Else active high
                        GPIO.setup(pins[key], GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Use pull-down

def _read_switch(pin: Optional[int]) -> bool:                   # Read a switch and return True if active
    if not USE_LIMIT_SWITCHES or pin is None:                   # If switches disabled or pin absent
        return False                                            # Treat as inactive
    val = GPIO.input(pin)                                       # Read GPIO level
    return (val == GPIO.LOW) if LIMIT_ACTIVE_LOW else (val == GPIO.HIGH)  # Interpret by logic mode

def _step_pulse(step_pin: int, period_s: float) -> None:        # Emit one STEP pulse with given period
    half = period_s / 2.0                                       # Half-period for HIGH and LOW
    GPIO.output(step_pin, GPIO.HIGH)                            # Drive STEP high
    time.sleep(half)                                            # Hold for half period
    GPIO.output(step_pin, GPIO.LOW)                             # Drive STEP low
    time.sleep(half)                                            # Hold for remaining half

def _dir_level(forward: bool) -> int:                           # Map boolean forward to GPIO level
    return GPIO.HIGH if forward else GPIO.LOW                   # HIGH = forward, LOW = reverse (invert if needed)

def _trapezoid_profile(total_steps: int) -> Tuple[int, int]:    # Compute accel & cruise span for trapezoid
    accel_steps_est = max(1, int((MIN_STEP_FREQ + MAX_STEP_FREQ) / 2.0 * ACCEL_TIME))  # Estimate accel steps
    accel = min(accel_steps_est, total_steps // 2)              # Bound accel to half of move
    cruise = max(0, total_steps - 2 * accel)                    # Remainder is cruise section
    return accel, cruise                                        # Return plan

def _freq_at_phase(i: int, accel_steps: int, cruise_steps: int, total_steps: int) -> float:  # Frequency for step i
    if i < accel_steps:                                         # Accelerating region
        return MIN_STEP_FREQ + (MAX_STEP_FREQ - MIN_STEP_FREQ) * (i / max(1, accel_steps))  # Linear ramp up
    elif i < accel_steps + cruise_steps:                        # Cruise region
        return MAX_STEP_FREQ                                    # Constant max frequency
    else:                                                       # Decelerating region
        j = i - (accel_steps + cruise_steps)                    # Index into decel phase
        return MAX_STEP_FREQ - (MAX_STEP_FREQ - MIN_STEP_FREQ) * (j / max(1, accel_steps))  # Ramp down

# ================================
# ---- One axis abstraction    ----
# ================================

class StepperAxis:                                              # Encapsulates one motor/axis behavior
    def __init__(self, name: str, pins: dict, steps_per_degree: float,
                 min_deg: float, max_deg: float):               # Constructor with config
        self.name = name                                        # Store axis name (AZ / EL)
        self.pins = pins                                        # Store pin map
        self.spd = steps_per_degree                             # Steps per degree at panel
        self.min_deg = min_deg                                  # Soft min angle
        self.max_deg = max_deg                                  # Soft max angle
        self.angle_deg = 0.0                                    # Current logical angle (deg)
        self.last_dir_positive = True                           # Track last direction for backlash logic
        GPIO.output(self.pins["EN"], GPIO.LOW)                  # Enable driver (LOW assumed active)

    def _move_steps(self, steps: int, forward: bool, obey_limits: bool = True) -> None:  # Low-level move
        if steps == 0:                                          # If no steps requested
            return                                              # Nothing to do
        if steps < 0:                                           # If negative step count
            steps = -steps                                      # Flip to positive steps
            forward = not forward                               # And invert direction
        GPIO.output(self.pins["DIR"], _dir_level(forward))      # Set DIR pin to requested direction
        accel, cruise = _trapezoid_profile(steps)               # Plan trapezoid timing
        for i in range(steps):                                  # Loop for each microstep
            if obey_limits and USE_LIMIT_SWITCHES:              # If we should obey switches
                if forward and _read_switch(self.pins.get("MAX_SW", None)):  # If moving forward & max hit
                    print(f"[{self.name}] MAX switch hit; stopping.")        # Log hit
                    break                                       # Break out of loop
                if not forward and _read_switch(self.pins.get("MIN_SW", None)):  # If moving backward & min hit
                    print(f"[{self.name}] MIN switch hit; stopping.")        # Log hit
                    break                                       # Break out
            freq = max(MIN_STEP_FREQ, _freq_at_phase(i, accel, cruise, steps))  # Compute step frequency
            _step_pulse(self.pins["STEP"], 1.0 / freq)          # Emit one pulse at that frequency
        self.last_dir_positive = forward                        # Remember last move direction

    def move_by_degrees(self, delta_deg: float, obey_limits: bool = True, use_backlash: bool = True) -> None:
        if delta_deg == 0:                                      # If zero motion request
            return                                              # Do nothing
        target_deg = self.angle_deg + delta_deg                 # Compute naïve target
        target_deg = max(self.min_deg, min(self.max_deg, target_deg))  # Clamp target to soft limits
        planned_delta = target_deg - self.angle_deg             # Actual delta after clamping
        if planned_delta == 0:                                  # If clamped out
            print(f"[{self.name}] Command would exceed soft limits; ignoring.")  # Warn
            return                                              # Exit
        forward = planned_delta > 0                             # True if move is positive
        if use_backlash and (forward != self.last_dir_positive):  # If changing direction, apply backlash
            overshoot_deg = BACKLASH_DEG                        # Overshoot amount to approach from one side
            main_steps  = int(round(abs(planned_delta) * self.spd))  # Steps to target
            extra_steps = int(round(overshoot_deg * self.spd))       # Steps for overshoot
            self._move_steps(main_steps + extra_steps, forward, obey_limits)   # Move past target
            self._move_steps(extra_steps, not forward, obey_limits)           # Back off to settle approach
            self.angle_deg = target_deg                         # Update logical angle to true target
            self.last_dir_positive = True                       # We ended with positive approach
            return                                              # Done
        steps = int(round(abs(planned_delta) * self.spd))       # Steps for normal move
        self._move_steps(steps, forward, obey_limits)           # Execute move
        self.angle_deg = target_deg                             # Update logical angle

    def home(self, toward_positive: bool, with_switch: bool = True,
             bump_back_deg: float = HOME_BUMP_BACK_DEG, offset_deg: float = HOME_OFFSET_DEG) -> None:
        print(f"[{self.name}] Homing {'with' if with_switch else 'without'} switch ...")  # Log start
        saved_max = globals()["MAX_STEP_FREQ"]                  # Save current max speed
        globals()["MAX_STEP_FREQ"] = 500                        # Slow down for safe homing
        try:                                                    # Ensure we restore speed even on error
            if with_switch:                                     # If using a switch
                target_pin = self.pins["MAX_SW"] if toward_positive else self.pins["MIN_SW"]  # Pick right pin
                if target_pin is None:                          # If pin not defined
                    raise RuntimeError("No switch pin defined for this homing direction.")  # Error out
                t0 = time.time()                                # Start a timeout timer
                while True:                                     # Loop until switch trips or timeout
                    if _read_switch(target_pin):                # If switch active
                        break                                   # Stop loop
                    self._move_steps(1, toward_positive, obey_limits=False)  # Crawl one step at a time
                    if time.time() - t0 > 20:                   # If >20s elapsed
                        raise TimeoutError("Homing timeout — check wiring/switch.")  # Fail with hint
                self.move_by_degrees(                           # After contact, back off slightly
                    -bump_back_deg if toward_positive else bump_back_deg,
                    obey_limits=False, use_backlash=False
                )
            else:                                               # Soft-home (no switch)
                travel = SOFT_HOME_TRAVEL_DEG                   # Choose gentle travel distance
                self.move_by_degrees(                           # Push into mechanical stop
                    travel if toward_positive else -travel,
                    obey_limits=False, use_backlash=False
                )
                self.move_by_degrees(                           # Back off slightly
                    -bump_back_deg if toward_positive else bump_back_deg,
                    obey_limits=False, use_backlash=False
                )
            self.angle_deg = 0.0 + offset_deg                   # Set logical zero with offset if needed
            print(f"[{self.name}] Homed. Angle set to {self.angle_deg:.3f}°")  # Log success
        finally:                                                # Always restore
            globals()["MAX_STEP_FREQ"] = saved_max              # Restore original max speed

    def return_to_zero(self) -> None:                           # Go back to logical zero
        self.move_by_degrees(-self.angle_deg, obey_limits=True, use_backlash=True)  # Move negative of current pos

# ===================================
# ---- High-level tracker wrapper ----
# ===================================

class SolarTracker:                                             # Orchestrates both axes and UART init
    def __init__(self):                                         # Constructor
        _ensure_gpio_setup()                                    # Init GPIO once
        if ENABLE_UART:                                         # If UART enabled
            self._init_uart()                                   # Configure drivers over UART
        self._update_kinematics()                               # Recompute steps/degree (maybe changed via UART)
        self.az = StepperAxis("AZ", AZ_PINS, self.steps_per_degree, AZ_MIN_DEG, AZ_MAX_DEG)  # Create AZ axis
        self.el = StepperAxis("EL", EL_PINS, self.steps_per_degree, EL_MIN_DEG, EL_MAX_DEG)  # Create EL axis
        self._load_state()                                      # Restore last positions from disk if present

    def _init_uart(self) -> None:                               # Configure both drivers via UART
        try:                                                    # Try…except so motion still works if UART fails
            self.uart_az = TMC2209(UART_PORT, AZ_SLAVE_ADDR, UART_BAUD)  # Open UART for AZ
            self.uart_el = TMC2209(UART_PORT, EL_SLAVE_ADDR, UART_BAUD)  # Open UART for EL
            for drv in (self.uart_az, self.uart_el):            # For each driver
                drv.set_gconf(                                  # Set global config
                    stealthchop=ENABLE_STEALTHCHOP,
                    interpolate=ENABLE_INTERPOLATE
                )
                drv.set_current(IRUN, IHOLD, IHOLDDELAY)        # Set motor currents
                drv.set_microsteps(UART_MICROSTEPS)             # Set external microstepping (MRES)
                drv.set_pwmconf_default()                       # Set PWM/stealthChop defaults
            global MICROSTEPPING                                # We will overwrite global MICROSTEPPING
            MICROSTEPPING = UART_MICROSTEPS                     # Adopt the UART microstep setting
            print(f"[UART] Set microstepping to 1/{MICROSTEPPING}")  # Log the new setting
        except Exception as e:                                  # If anything goes wrong
            print("[UART] Skipping UART configuration (error). Reason:", e)  # Warn but continue

    def _update_kinematics(self) -> None:                       # Recompute steps/degree if microstepping changed
        self.steps_per_panel_rev = MOTOR_STEPS_PER_REV * MICROSTEPPING * GEAR_RATIO  # Recalculate steps/rev
        self.steps_per_degree    = self.steps_per_panel_rev / 360.0                  # Update steps/degree

    def _load_state(self) -> None:                              # Load last angles from disk
        if os.path.exists(STATE_PATH):                          # If file exists
            try:                                                # Try to read/parse
                with open(STATE_PATH, "r") as f:                # Open file
                    data = json.load(f)                         # Parse JSON
                self.az.angle_deg = float(data.get("az_deg", 0.0))  # Set AZ angle
                self.el.angle_deg = float(data.get("el_deg", 0.0))  # Set EL angle
                print(f"[STATE] Restored AZ={self.az.angle_deg:.3f}°, EL={self.el.angle_deg:.3f}°")  # Log restore
            except Exception as e:                              # If parse fails
                print("[STATE] Failed to load state; starting at 0/0. Err:", e)  # Warn
        else:                                                   # If no file
            pass                                                # Leave defaults at 0/0

    def _save_state(self) -> None:                              # Save current angles atomically
        data = {"az_deg": self.az.angle_deg, "el_deg": self.el.angle_deg}  # Build dictionary
        tmp = STATE_PATH + ".tmp"                               # Temp file path
        with open(tmp, "w") as f:                               # Open temp for writing
            json.dump(data, f, indent=2)                        # Write pretty JSON
        os.replace(tmp, STATE_PATH)                             # Atomic rename to final path

    # -------- Public API (what your AI and CLI will call) --------

    def home_to_start(self) -> None:                            # Home both axes to start
        GPIO.output(AZ_PINS["EN"], GPIO.LOW)                    # Ensure AZ enabled (LOW assumed active)
        GPIO.output(EL_PINS["EN"], GPIO.LOW)                    # Ensure EL enabled
        self.az.home(toward_positive=AZ_HOME_TOWARD_POSITIVE, with_switch=USE_LIMIT_SWITCHES)  # Home AZ
        self.el.home(toward_positive=EL_HOME_TOWARD_POSITIVE, with_switch=USE_LIMIT_SWITCHES)  # Home EL
        self._save_state()                                      # Persist angles

    def move_by(self, az_delta_deg: float, el_delta_deg: float) -> None:  # Relative move interface
        print(f"[CMD] move_by: dAZ={az_delta_deg:.3f}°, dEL={el_delta_deg:.3f}°")  # Log command
        self.az.move_by_degrees(az_delta_deg, obey_limits=True, use_backlash=True) # Move AZ
        self.el.move_by_degrees(el_delta_deg, obey_limits=True, use_backlash=True) # Move EL
        self._save_state()                                      # Save new angles

    def move_to(self, az_target_deg: float, el_target_deg: float) -> None:         # Absolute move helper
        self.move_by(az_target_deg - self.az.angle_deg,                              # Convert to relative AZ
                      el_target_deg - self.el.angle_deg)                             # Convert to relative EL

    def return_to_start_of_day(self) -> None:                     # End-of-day routine
        print("[CMD] return_to_start_of_day")                     # Log action
        self.az.return_to_zero()                                  # Zero AZ
        self.el.return_to_zero()                                  # Zero EL
        self._save_state()                                        # Persist zeros

    def jog(self, which: str, step_deg: float, count: int = 1, delay_s: float = 0.1) -> None:
        """
        Jog helper for bench testing and small adjustments.

        which: 'az', 'el', or 'both'
        step_deg: signed degrees per jog (e.g., +0.2 or -0.5)
        count: how many times to repeat the jog (>=1)
        delay_s: pause between jogs (seconds)
        """
        print(f"[CMD] jog: which={which}, step={step_deg}°, count={count}, delay={delay_s}s")  # Log jog command
        count = max(1, int(count))                                   # Ensure count is at least 1
        for i in range(count):                                       # Repeat jogging 'count' times
            if which.lower() == "az":                                # Jog azimuth only
                self.move_by(step_deg, 0.0)                          # Move AZ by step, EL unchanged
            elif which.lower() == "el":                              # Jog elevation only
                self.move_by(0.0, step_deg)                          # Move EL by step, AZ unchanged
            elif which.lower() == "both":                            # Jog both axes
                self.move_by(step_deg, step_deg)                     # Move both by same step
            else:                                                    # Invalid axis name
                raise ValueError("which must be 'az', 'el', or 'both'")  # Inform caller
            if i < count - 1:                                        # If more jogs remain
                time.sleep(max(0.0, delay_s))                        # Pause between jogs

    def shutdown(self) -> None:                                   # Clean shutdown
        try:                                                      # Ensure we always cleanup GPIO
            GPIO.output(AZ_PINS["EN"], GPIO.HIGH)                 # Disable AZ (HIGH assumed disable)
            GPIO.output(EL_PINS["EN"], GPIO.HIGH)                 # Disable EL
        finally:                                                  # Finally block guarantees cleanup
            GPIO.cleanup()                                        # Release GPIO resources

# ======================================
# ---- Command Line Interface (CLI)  ----
# ======================================

if __name__ == "__main__":                        # Runs only if file is executed directly
    import argparse                               # Library for parsing command-line arguments

    # ---------- Argument parser ----------
    parser = argparse.ArgumentParser(
        description="Dual-axis solar tracker controller (Raspberry Pi + TMC2209)."  # CLI description
    )

    # ---- Global options that override config ----
    parser.add_argument("--uart", dest="uart", action="store_true",                # Flag to force-enable UART
                        help="Enable UART configuration (overrides ENABLE_UART).")
    parser.add_argument("--no-uart", dest="uart", action="store_false",            # Flag to force-disable UART
                        help="Disable UART configuration at startup.")
    parser.set_defaults(uart=None)                # Default: None means use file’s ENABLE_UART

    parser.add_argument("--microsteps", type=int,                                   # Desired microsteps via UART
                        choices=[256,128,64,32,16,8,4,2,1],
                        help="Set microstepping (UART).")
    parser.add_argument("--irun", type=int, help="Running current (0..31).")       # Run current override
    parser.add_argument("--ihold", type=int, help="Holding current (0..31).")      # Hold current override
    parser.add_argument("--iholddelay", type=int, help="Hold current delay (0..15).")  # IHOLDDELAY override

    parser.add_argument("--stealthchop", dest="stealthchop", action="store_true",  # Enable stealthChop
                        help="Enable stealthChop mode.")
    parser.add_argument("--no-stealthchop", dest="stealthchop", action="store_false",  # Disable stealthChop
                        help="Disable stealthChop (use spreadCycle).")
    parser.set_defaults(stealthchop=None)       # None keeps file default

    parser.add_argument("--interpolate", dest="interpolate", action="store_true",  # Enable 256x interpolation
                        help="Enable interpolation to 256 µsteps.")
    parser.add_argument("--no-interpolate", dest="interpolate", action="store_false",  # Disable interpolation
                        help="Disable interpolation.")
    parser.set_defaults(interpolate=None)      # None keeps file default

    # ---- Subcommands (actual actions) ----
    sub = parser.add_subparsers(dest="cmd", required=True)      # Create subcommand group (required)

    # home
    sub.add_parser("home", help="Home both axes to starting position.")            # 'home' subcommand

    # move-by
    p_move_by = sub.add_parser("move-by", help="Relative move by dAZ dEL (degrees).")  # 'move-by' parser
    p_move_by.add_argument("daz", type=float, help="Delta azimuth (deg).")         # First positional arg
    p_move_by.add_argument("del_", type=float, help="Delta elevation (deg).")      # Second positional arg

    # move-to
    p_move_to = sub.add_parser("move-to", help="Absolute move to AZ EL (degrees).")    # 'move-to' parser
    p_move_to.add_argument("az", type=float, help="Target azimuth (deg).")         # Absolute AZ
    p_move_to.add_argument("el", type=float, help="Target elevation (deg).")       # Absolute EL

    # return
    sub.add_parser("return", help="Return both axes to starting position (end of day).")  # 'return' command

    # status
    sub.add_parser("status", help="Print current angles and configuration.")        # 'status' command

    # jog
    p_jog = sub.add_parser("jog", help="Jog axis by a small step repeatedly for testing/adjustments.")  # 'jog' cmd
    p_jog.add_argument("which", choices=["az", "el", "both"],                       # Which axis to jog
                       help="Which axis to jog: 'az', 'el', or 'both'.")
    p_jog.add_argument("step", type=float,                                        # Step size in degrees
                       help="Signed step in degrees per jog (e.g., 0.2 or -0.5).")
    p_jog.add_argument("--count", type=int, default=1,                            # How many jogs to perform
                       help="Number of jog repetitions (default: 1).")
    p_jog.add_argument("--delay", type=float, default=0.1,                        # Delay between jogs
                       help="Delay between jogs in seconds (default: 0.1).")

    args = parser.parse_args()                        # Parse CLI arguments into `args`

    # ---------- Apply CLI overrides ----------
    # These replace the global constants before SolarTracker is created
    if args.uart is not None:
        ENABLE_UART = bool(args.uart)                  # Override UART enable/disable
    if args.microsteps is not None:
        UART_MICROSTEPS = args.microsteps              # Override desired UART microsteps
    if args.irun is not None:
        IRUN = max(0, min(31, args.irun))              # Clamp IRUN to valid range
    if args.ihold is not None:
        IHOLD = max(0, min(31, args.ihold))            # Clamp IHOLD to valid range
    if args.iholddelay is not None:
        IHOLDDELAY = max(0, min(15, args.iholddelay))  # Clamp IHOLDDELAY to valid range
    if args.stealthchop is not None:
        ENABLE_STEALTHCHOP = bool(args.stealthchop)    # Override stealthChop enable
    if args.interpolate is not None:
        ENABLE_INTERPOLATE = bool(args.interpolate)    # Override interpolation enable

    # ---------- Run command ----------
    tracker = SolarTracker()                      # Create tracker object (sets up GPIO + UART + axes)
    try:
        if args.cmd == "home":                    # If user requested "home"
            tracker.home_to_start()               # Perform homing

        elif args.cmd == "move-by":               # If "move-by"
            tracker.move_by(args.daz, args.del_)  # Pass delta azimuth/elevation

        elif args.cmd == "move-to":               # If "move-to"
            tracker.move_to(args.az, args.el)     # Move to absolute AZ/EL

        elif args.cmd == "return":                # If "return"
            tracker.return_to_start_of_day()      # Go back to logical zero

        elif args.cmd == "status":                # If "status"
            # Print current angles + config info
            print(f"AZ angle: {tracker.az.angle_deg:.3f}°")                                 # Report AZ
            print(f"EL angle: {tracker.el.angle_deg:.3f}°")                                 # Report EL
            print(f"UART enabled: {ENABLE_UART}")                                           # Report UART flag
            print(f"Microsteps: 1/{MICROSTEPPING} (target via UART: 1/{UART_MICROSTEPS})")  # Report µsteps
            print(f"Currents: IRUN={IRUN}, IHOLD={IHOLD}, IHOLDDELAY={IHOLDDELAY}")         # Report currents
            print(f"Modes: stealthChop={'on' if ENABLE_STEALTHCHOP else 'off'}, "
                  f"interpolate={'on' if ENABLE_INTERPOLATE else 'off'}")                   # Report modes

        elif args.cmd == "jog":                                                             # If "jog"
            tracker.jog(args.which, args.step, count=args.count, delay_s=args.delay)        # Execute jog routine

    finally:
        tracker.shutdown()                        # Always disable motors + cleanup GPIO





# python3 solar_tracker.py home
# python3 solar_tracker.py move-by 2 -1.5
# python3 solar_tracker.py move-to 15 25
# python3 solar_tracker.py return
# python3 solar_tracker.py status
# python3 solar_tracker.py jog az 0.2 --count 10 --delay 0.15
# python3 solar_tracker.py --microsteps 256 --irun 20 --ihold 8 home
# python3 solar_tracker.py --no-uart move-by -0.25 0.25