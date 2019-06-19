import struct
import numpy as np 


def generate_1d_sine(points = 1000):
    x = np.linspace(0, 20, points)
    y = np.sin(x) + (np.random.randn(points) * 0.1)
    return y

def generate_2d():
    data = np.zeros((3, 3))
    for i in range(3):
        data[i, :] = generate_1d_sine(3)
    return data

def load_assi_data(filename):
    with open(filename, 'r') as stream:
        ver_num = stream.readline().strip()
        ManufacID = stream.readline().strip()
        CreateDate = stream.readline().strip()
        ModDate = stream.readline().strip()
        NumPoints = stream.readline().strip()           # columns
        n_columns = int(NumPoints.split('=')[1])
        NumProfiles = stream.readline().strip()         # rows
        n_rows = int(NumProfiles.split('=')[1])
        Xscale = stream.readline().strip()
        Yscale = stream.readline().strip()
        Zscale = stream.readline().strip()
        Zresolution = stream.readline().strip()
        Compression = stream.readline().strip()
        DataType = stream.readline().strip()
        print("DataType: %s" % DataType)
        CheckType = stream.readline().strip()
        stream.readline()
        stream.readline()

        data = []
        for line in stream:
            strs = line.strip().split(' ')
            if len(strs) < n_columns:
                break
            row = []
            for item in strs:
                row.append(float(item))
            data.append(row)
        return data

def load_data(filename):
    with open(filename, 'rb') as stream:
        # SDF Format: https://physics.nist.gov/VSC/Help/DataFormat/SDF_Format.jsp
        ver_num = stream.read(8)
        ManufacID = stream.read(10)
        CreateDate = stream.read(12)
        ModDate = stream.read(12)
        NumPoints = stream.read(2)  # columns
        NumProfiles = stream.read(2) # rows
        Xscale = stream.read(8)
        Yscale = stream.read(8)
        Zscale = stream.read(8)
        Zresolution = stream.read(8)
        Compression = stream.read(1)
        DataType = stream.read(1)
        CheckType = stream.read(1)

        n_columns,  = struct.unpack('h', NumPoints)   
        n_rows,  = struct.unpack('h', NumProfiles)
        col_scale, = struct.unpack('d', Xscale)
        row_scale, = struct.unpack('d', Yscale) 
        z_scale, = struct.unpack('d', Zscale)
        z_res, = struct.unpack('d', Zresolution) 
        data_type, = struct.unpack('b', DataType)

        # print('''%s
        #          ManufacID            = %s
        #          CreateDate            = %s
        #          ModDate               = %s
        #          NumPoints             = %s
        #          NumProfiles           =  %s
        #          Xscale                   = %s
        #          Yscale                   = %s
        #          Zscale                   = %s
        #          Zresolution            = %s
        #          Compression         = %s
        #          DataType             = %s
        #          CheckType          = %s''' % (ver_num, ManufacID, CreateDate, ModDate, NumPoints, NumProfiles, Xscale, Yscale, Zscale, Zresolution, Compression, DataType, CheckType))

        print('data size: %d * %d; Zresolution: %e' % (n_rows, n_columns, z_scale))
        data = np.zeros((n_rows, n_columns), dtype=np.float)
        for irow in range(n_rows):
            for icol in range(n_columns):
                temp = stream.read(4)
                itemp, = struct.unpack('i', temp)
                data[irow, icol] = itemp * z_scale
    return data

def load_measured_data():
    filename = "data/DemoSteel.sdf"
    # filename = "data/HipHeaddemo.sdf"
    data = load_data(filename)
    return data

if __name__ == '__main__':
    from visualization import *
    data = generate_2d()
    show3d_surface(data)
    waitkey()