import time

class Runtime:
    
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None

    def start(self):
        self.start_time = time.time()
        self.end_time = None

    def record_seconds(self):
        now = time.time()
        seconds = int(now - self.start_time)
        return seconds

    def record(self):
        seconds = self.record_seconds()
        secs = seconds % 60
        mins = seconds / 60 % 60
        hours = seconds / 60 / 60
        
        return hours, mins, secs

    def end(self):
        self.end_time = time.time()

    def total_seconds(self):
        seconds = int(self.end_time - self.start_time)
        return seconds

    def total(self):
        seconds = self.total_seconds()
        secs = seconds % 60
        mins = seconds / 60 % 60
        hours = seconds / 60 / 60
        
        return hours, mins, secs

    def desc(self):
        if self.end is None:
            hours, mins, secs = self.record()
        else:
            hours, mins, secs = self.total()
        
        return '%d:%d:%d' % (hours, mins, secs)

