class GridManagerAgent:

    def __init__(self, capacity_limit_ratio=0.9):
        self.capacity_limit_ratio = capacity_limit_ratio

    def apply_grid_control(self, building_agents):

        total_load = sum(agent.adjusted_load for agent in building_agents)
        capacity_limit = total_load * self.capacity_limit_ratio
        overload = total_load - capacity_limit

        if overload <= 0:
            return total_load, total_load

        # Sort by priority (low priority curtailed first)
        building_agents.sort(
            key=lambda x: x.priority_level,
            reverse=True
        )

        for agent in building_agents:
            if overload <= 0:
                break
            reduced = agent.curtail_load(overload)
            overload -= reduced

        adjusted_total = sum(agent.adjusted_load for agent in building_agents)

        return total_load, adjusted_total