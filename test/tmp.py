import os

def dataHandle(file_path):
    outLines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(' ')
            data[0] = data[0] + ','
            new_line = ' '.join(data) + '\n'
            outLines.append(new_line)
    
    output_path = file_path[:-4] + '.csv'
    with open(output_path, 'w') as fw:x1
        fw.writelines(outLines)


for dataFile in os.listdir("/data/zhr_data/AutoGraph/test"):
    if dataFile.endswith('.txt'):
        input_file = os.path.join("/data/zhr_data/AutoGraph/test", dataFile)
        dataHandle(input_file)