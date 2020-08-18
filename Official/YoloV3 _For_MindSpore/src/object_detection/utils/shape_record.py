class Shape_record(object):
    def __init__(self):
        self.shape_record = {
            320: 0,
            352: 0,
            384: 0,
            416: 0,
            448: 0,
            480: 0,
            512: 0,
            544: 0,
            576: 0,
            608: 0,
            'total': 0
        }

    def set(self, shape):
        if len(shape) > 1:
            shape = shape[0]
        shape = int(shape)
        self.shape_record[shape]+=1
        self.shape_record['total'] +=1

    def show(self, logger):
        for key in self.shape_record:
            rate = self.shape_record[key] / float(self.shape_record['total'])
            logger.info('shape {}: {:.2f}%'.format(key, rate*100))
            #print('shape {}: {:.2f}%'.format(key, rate*100))





