from dataclasses import dataclass


@dataclass
class Battery:
    capacity_kwh: float = 100.0      # Max capacity
    current_soc_kwh: float = 50.0    # Current State of Charge
    max_charge_kw: float = 20.0     # Max charge rate
    max_discharge_kw: float = 20.0  # Max discharge rate
    efficiency: float = 0.95        # Charge/discharge efficiency multiplier

    def step(
        self,
        action_kw: float,
        duration_hours: float = 1.0
       ) -> float:
        """
        Applies charge/discharge action and returns the net change in SoC (kWh).

        Args:
            action_kw: Requested power flow, in kW (positive for charge, negative for discharge, null for holding).
            duration_hours: Time step duration, in hours.

        Returns:
            Actual change in stored energy (kWh). Positive if charged, negative if discharged, null if held.
        """

        # Charging
        if action_kw > 0:
            # Enforce max charge rate constraint (kW)
            effective_power_kw = min(action_kw, self.max_charge_kw)

            # Energy attempting to enter the chemistry (kWh)
            gross_energy_kwh = effective_power_kw * duration_hours * self.efficiency

            # Cap by remaining headroom in battery
            available_headroom_kwh = self.capacity_kwh - self.current_soc_kwh
            delta_soc_kwh = min(gross_energy_kwh, available_headroom_kwh)

        # Discharging
        elif action_kw < 0:
            # Enforce max discharge rate constraint (kW)
            requested_discharge_kw = abs(action_kw)
            effective_power_kw = min(requested_discharge_kw, self.max_discharge_kw)

            # Energy drained from chemistry accounting for efficiency loss (kWh)
            gross_energy_kwh = (effective_power_kw * duration_hours) / self.efficiency

            # Cap by available energy in battery
            delta_soc_kwh = -min(gross_energy_kwh, self.current_soc_kwh)

        # Holding
        else:
            delta_soc_kwh = 0.0

        # Update internal state
        self.current_soc_kwh += delta_soc_kwh

        return delta_soc_kwh
