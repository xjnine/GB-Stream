import pandas as pd

from models.MGBStream import start

if __name__ == '__main__':
    recordFileName = "RBF3(1000).csv"
    # recordFileName = "star(200).csv"
    # wafer 300; star 400; cover 1000
    # csv = pd.read_csv("data/real/StarLightCurves.csv", header=None)
    csv = pd.read_csv("data/synthetic/RBF3_40000(1000).csv", header=None)
    print(csv)
    start(csv, recordFileName, v=1000)
    print(recordFileName + "end")

    recordFileName = "RBF3(1200).csv"
    csv = pd.read_csv("data/synthetic/RBF3_40000(1000).csv", header=None)
    print(csv)
    start(csv, recordFileName, v=1200)
    print(recordFileName + "end")

    recordFileName = "RBF3(1400).csv"
    csv = pd.read_csv("data/synthetic/RBF3_40000(1000).csv", header=None)
    print(csv)
    start(csv, recordFileName, v=1400)
    print(recordFileName + "end")

    recordFileName = "RBF3(1600).csv"
    csv = pd.read_csv("data/synthetic/RBF3_40000(1000).csv", header=None)
    print(csv)
    start(csv, recordFileName, v=1600)
    print(recordFileName + "end")

    recordFileName = "RBF3(1800).csv"
    csv = pd.read_csv("data/synthetic/RBF3_40000(1000).csv", header=None)
    print(csv)
    start(csv, recordFileName, v=1800)
    print(recordFileName + "end")
