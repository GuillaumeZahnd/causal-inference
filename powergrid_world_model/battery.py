from dataclasses import dataclass
from typing import Tuple


@dataclass
class Battery:
    capacity_kwh: float = 100.0     # Max capacity
    current_soc_kwh: float = 50.0   # Current State of Charge
    max_charge_kw: float = 20.0     # Max charge rate
    max_discharge_kw: float = 20.0  # Max discharge rate
    efficiency: float = 0.95        # Charge/discharge efficiency multiplier

    @staticmethod
    def compute_step(
        soc_kwh: float,
        action_kw: float,
        duration_hours: float,
        capacity_kwh: float,
        max_charge_kw: float,
        max_discharge_kw: float,
        efficiency: float,
    ) -> Tuple[float, float]:
        """
        Pure physics computation for a single step.
        Calculates physical changes without mutating state.
        """
        if duration_hours <= 0:
            return 0.0, 0.0

        # Charging (drawing power from the grid)
        if action_kw > 0:

            # Power we are trying to get from the grid to the battery (action, capped by the maximum charge rate)
            effective_power_kw = min(action_kw, max_charge_kw)

            # Energy we are trying to get from grid to battery (power multiplied by time, with an efficiency tax)
            gross_energy_kwh = effective_power_kw * duration_hours * efficiency

            # The battery can currently accept this much energy
            available_headroom_kwh = capacity_kwh - soc_kwh

            # This much energy is added to the battery from the grid
            delta_soc_kwh = min(gross_energy_kwh, available_headroom_kwh)

            if gross_energy_kwh > 0:
                # This much power left the grid to charge the battery
                delta_grid_kw = delta_soc_kwh / (duration_hours * efficiency)
            else:
                # No power left the grid to charge the battery, because the battery was already full
                delta_grid_kw = 0.0

        # Discharging (adding power to the grid)
        elif action_kw < 0:

            # Power we are trying to send from the battery to the grid (action, capped by the maximum discharge rate)
            requested_discharge_kw = abs(action_kw)
            effective_power_kw = min(requested_discharge_kw, max_discharge_kw)

            # Energy we need to pull from the battery to hit that power (power multiplied by time, with an efficiency tax)
            gross_energy_kwh = (effective_power_kw * duration_hours) / efficiency

            # The battery can currently supply this much energy
            available_charge_kwh = soc_kwh

            # This much energy is taken from the battery to send to the grid
            max_drain_kwh = min(gross_energy_kwh, available_charge_kwh)
            delta_soc_kwh = -max_drain_kwh

            if gross_energy_kwh > 0:
                # This much power was delivered to the grid from the battery (negative means export)
                delta_grid_kw = -(max_drain_kwh * efficiency) / duration_hours
            else:
                # No power was sent to the grid, because the battery was already empty
                delta_grid_kw = 0.0

        # Holding
        else:
            delta_soc_kwh = 0.0
            delta_grid_kw = 0.0

        return delta_soc_kwh, delta_grid_kw

    def step(
        self,
        action_kw: float,
        duration_hours: float,
    ) -> Tuple[float, float]:
        """
        Apply charge/discharge action and returns physical battery changes.

        Args:
            action_kw: Requested power flow in kW (positive: charging, negative: discharging, null: holding).
            duration_hours: Time step duration in hours.

        Returns:
            Tuple of (delta_soc_kwh, delta_grid_kw):
                - delta_soc_kwh: Actual energy change in battery (+ for charge, - for discharge).
                - delta_grid_kw: Actual power flow at grid connection (+ for import, - for export).
        """
        # Compute pure physics step
        delta_soc_kwh, delta_grid_kw = self.compute_step(
            soc_kwh=self.current_soc_kwh,
            action_kw=action_kw,
            duration_hours=duration_hours,
            capacity_kwh=self.capacity_kwh,
            max_charge_kw=self.max_charge_kw,
            max_discharge_kw=self.max_discharge_kw,
            efficiency=self.efficiency,
        )

        # Update internal state
        self.current_soc_kwh += delta_soc_kwh

        return delta_soc_kwh, delta_grid_kw
