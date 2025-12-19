
import os

def create_directories(base_path, count=20):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i in range(count):
        inner_path = os.path.join(base_path, "dir_" + str(i))
        if not os.path.exists(inner_path):
            os.makedirs(inner_path)


if __name__ == "__main__":
    path = r"C:\Users\Lenovo\Desktop\DEPI(R_2)\ML_git\GIZ4_AIS2_S1_Ml\src\Python"
    create_directories(path)