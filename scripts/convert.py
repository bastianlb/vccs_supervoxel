import numpy as np

if __name__ == "__main__":
    data = np.load("test_data/0566_00.npy")
    with open('scannet.xyz', 'w+') as f:
        for i in range(data.shape[0]):
            f.write(" ".join([str(x) for x in data[i, :3]]))
            f.write(" ")
            f.write(" ".join([str(int(x)) for x in data[i, 3:6]]))
            f.write("\n")

