from datetime import datetime
from typing import Optional
import time
from pytz import UTC

import omni.ext
import omni.ui as ui
import omni.timeline
from omni.timeline import TimelineEventType
import carb.events
from omni.kit.viewport.utility import get_active_viewport_window

import omni.kit.pipapi

from .style import _PAUSE, _PLAY, clock_box, clock_label, SIMULATION_CONTROLS

omni.kit.pipapi.install("skyfield")
from skyfield.api import load, Time

class TimeControlFrame(ui.Frame):

    def __init__(self):
        super().__init__(spacing=0, style=SIMULATION_CONTROLS)

        self.time_controller = SimulationTimeController()

        # UI models to store the values from the input fields
        now = datetime.now(UTC)
        self.year_model = ui.SimpleIntModel(now.year)
        self.month_model = ui.SimpleIntModel(now.month)
        self.day_model = ui.SimpleIntModel(now.day)
        self.hour_model = ui.SimpleIntModel(now.hour)
        self.minute_model = ui.SimpleIntModel(now.minute)
        self.second_model = ui.SimpleIntModel(now.second)
        self.time_scale_model = ui.SimpleFloatModel(SimulationTimeController.DEFAULT_SPEED)

        self._window: Optional[ui.Window] = get_active_viewport_window()
        self._margin = 20

        self._timeline = omni.timeline.get_timeline_interface()
        timeline_event_stream = self._timeline.get_timeline_event_stream()
        self._prev_time = time.time()
        self._subscription = timeline_event_stream.create_subscription_to_pop_by_type(
                int(TimelineEventType.CURRENT_TIME_TICKED),
                self._on_update
            )

        self._datetime_fields: list[ui.IntField] = []

        self.set_build_fn(self._build_fn)

    def _build_fn(self):
        w, h = 300, 100
        with ui.Placer(offset_x=self._window.width * 0.5 - w * 0.65, offset_y=self._window.height - self._margin - h - 25):
                with ui.ZStack(width=w, height=h, content_clipping=1, opaque_for_mouse_events=True):
                    ui.Rectangle(width=ui.Percent(100), height=ui.Percent(100))
                    with ui.VStack(spacing=0):

                        # --- SECTION FOR SIMULATION CONTROL ---
                        with ui.HStack(height=30, spacing=0):
                            # play/pause button
                            btn = ui.Button(
                                "",
                                tooltip='Play/Pause Simulation',
                                width=0,
                                height=0,
                                image_width=25,
                                image_height=25,
                                image_url = _PAUSE,
                            )
                            btn.set_clicked_fn(lambda b=btn: self._on_play_pause_click(b))

                            def build_year_box(model):
                                o = ui.IntField(model, enabled=False, width=50, style=clock_box)
                                self._datetime_fields.append(o)
                            def build_date_box(model):
                                o = ui.IntField(model, enabled=False, width=30, style=clock_box)
                                self._datetime_fields.append(o)
                            def build_time_box(model):
                                o = ui.IntField(model, enabled=False, width=30, style=clock_box)
                                self._datetime_fields.append(o)

                            ui.Label("Time:")
                            build_date_box(self.month_model)
                            ui.Label("/", style=clock_label, width=3)
                            build_date_box(self.day_model)
                            ui.Label("/", style=clock_label, width=3)
                            build_year_box(self.year_model)
                            ui.Spacer(width=10)
                            build_time_box(self.hour_model)
                            ui.Label(":", style=clock_label, width=3)
                            build_time_box(self.minute_model)
                            ui.Label(":", style=clock_label, width=3)
                            build_time_box(self.second_model)

                        ui.Spacer()
                        with ui.HStack(height=30, spacing=3):

                            ui.Label("Speed:")
                            # A slider for more intuitive speed control
                            ui.FloatSlider(self.time_scale_model, min=SimulationTimeController.MIN_SPEED,
                                           max=SimulationTimeController.MAX_SPEED, step=0.1,
                                            format="%.1f x", width=ui.Percent(80))
                            self.time_scale_model.add_value_changed_fn(
                                lambda m: self.time_controller.set_time_scale(m.get_value_as_float())
                            )

    def _on_play_pause_click(self, btn: ui.Button):
        # Make this a toggle checkbox
        #btn.checked = not btn.checked

        #if btn.checked:
        if not self.time_controller._is_running:
            btn.image_url = _PAUSE
            # Make fields uneditable
            for dt_field in self._datetime_fields:
                dt_field.enabled = False

            self.time_controller.set_time(
                self.year_model.get_value_as_int(),
                self.month_model.get_value_as_int(),
                self.day_model.get_value_as_int(),
                self.hour_model.get_value_as_int(),
                self.minute_model.get_value_as_int(),
                self.second_model.get_value_as_int()
            )

            self.time_controller.play()
        else:
            btn.image_url = _PLAY
            # Make fields editable
            for dt_field in self._datetime_fields:
                dt_field.enabled = True
            self.time_controller.pause()

    def _on_set_time_click(self):
        """Called when the 'Apply Date and Time' button is clicked."""
        self.time_controller.set_time(
            self.year_model.get_value_as_int(),
            self.month_model.get_value_as_int(),
            self.day_model.get_value_as_int(),
            self.hour_model.get_value_as_int(),
            self.minute_model.get_value_as_int(),
        )

    def _on_update(self, e: carb.events.IEvent):
        """This function is called every single frame."""
        t = time.time()
        delta_time = t - self._prev_time
        self._prev_time = t

        # 1. Advance the time in our controller
        self.time_controller.update(delta_time)

        # 2. Get the new current time from the controller
        current_dt = self.time_controller.get_current_datetime()

        if self.time_controller._is_running:
            # 3. Update current time models
            self.year_model.set_value(current_dt.year)
            self.month_model.set_value(current_dt.month)
            self.day_model.set_value(current_dt.day)
            self.hour_model.set_value(current_dt.hour)
            self.minute_model.set_value(current_dt.minute)
            self.second_model.set_value(current_dt.second)


class SimulationTimeController:
    """
    Manages the state of the simulation time, including date, speed, and play/pause.
    """

    DEFAULT_SPEED = 1.0
    MIN_SPEED = 0.1
    MAX_SPEED = 50.0

    def __init__(self):
        """Initializes the time controller."""
        self.ts = load.timescale()
        self._is_running = True  # Start the simulation in a 'playing' state
        self._time_scale = SimulationTimeController.DEFAULT_SPEED   # 1.0 means real-time. > 1.0 is fast-forward.
        self._current_sim_time = self.ts.now() # The current time of the simulation

        global _time_controller
        _time_controller = self

    def update(self, delta_time: float):
        """
        This method should be called every frame to advance the simulation time.

        Args:
            delta_time (float): The real-world time in seconds that has passed since the last frame.
        """

        if not self._is_running:
            return

        # Calculate how much simulation time has passed
        # Skyfield's time is measured in days, so we convert delta_time from seconds to days.
        simulation_seconds_passed = delta_time * self._time_scale
        simulation_days_passed = simulation_seconds_passed / 86400.0  # (24 * 60 * 60)

        # Advance the current simulation time
        self._current_sim_time = self.ts.tt_jd(self._current_sim_time.tt + simulation_days_passed)

        from .extension import get_sim_manager
        dt = self.get_current_datetime()
        get_sim_manager()._on_timestep(dt)

    def set_time(self, year: int, month: int, day: int, hour: int = 12, minute: int = 0, second: int = 0):
        """
        Manually sets the simulation to a specific UTC date and time.
        """
        try:
            self._current_sim_time = self.ts.utc(year, month, day, hour, minute, second)
            print(f"Simulation time set to: {self._current_sim_time.utc_iso()}")
        except ValueError as e:
            print(f"Error setting date: {e}. Please provide a valid date.")

    def play(self):
        """Starts or resumes the simulation."""
        self._is_running = True

    def pause(self):
        """Pauses the simulation."""
        self._is_running = False

    def set_time_scale(self, scale: float):
        """
        Sets the speed of the simulation.
        1.0 = real-time
        60.0 = 1 minute of simulation time per real second
        3600.0 = 1 hour of simulation time per real second
        """
        self._time_scale = max(0, scale) # Prevent negative time scales

    def get_current_time(self) -> Time:
        """Returns the current Skyfield time object for the simulation."""
        return self._current_sim_time

    def get_current_datetime(self) -> datetime:
        """Returns the current python DateTime object for the simulation in UTC."""
        return self._current_sim_time.utc_datetime()
