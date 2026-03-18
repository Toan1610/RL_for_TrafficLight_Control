
class CollectedData:
    def __init__(self):
        self.list_occupancy = []
        self.average_occupancy = 0.0
    
    def update(self, occupancy):
        self.list_occupancy.append(occupancy)
        if self.list_occupancy:
            self.average_occupancy = sum(self.list_occupancy) / len(self.list_occupancy)
        else:
            self.average_occupancy = 0.0
        
    


