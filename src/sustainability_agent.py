class SustainabilityAgent:

    def summarize(self, before_loads, after_loads):

        peak_before = max(before_loads)
        peak_after = max(after_loads)

        total_before = sum(before_loads)
        total_after = sum(after_loads)

        reduction_percentage = (
            (total_before - total_after) / total_before
        ) * 100 if total_before > 0 else 0

        peak_smoothing = (
            (peak_before - peak_after) / peak_before
        ) * 100 if peak_before > 0 else 0

        co2_reduction = (total_before - total_after) * 0.4

        return {
            "Peak Before": peak_before,
            "Peak After": peak_after,
            "Peak Reduction (%)": peak_smoothing,
            "Total Reduction (%)": reduction_percentage,
            "Estimated CO2 Reduction (kg)": co2_reduction
        }