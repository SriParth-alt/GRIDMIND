class BuildingEnergyAgent:

    def __init__(self, building_id, model, priority_level=1):
        self.building_id = building_id
        self.model = model
        self.priority_level = priority_level  # 1 = high priority, 3 = low priority
        self.predicted_load = 0
        self.adjusted_load = 0
        self.total_curtailed = 0

    def predict_load(self, feature_row):
        self.predicted_load = self.model.predict(feature_row)[0]
        self.adjusted_load = self.predicted_load
        return self.predicted_load

    def curtail_load(self, reduction_amount):
        reduction = min(reduction_amount, self.adjusted_load * 0.3)
        self.adjusted_load -= reduction
        self.total_curtailed += reduction
        return reduction