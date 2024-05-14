from multiprocessing import Pool

def changeText(args):
    txtList, id = args
    txtList[id] = 'Id:'+str(id)

def giveText(args):
    id = args[0]
    return str(id)

if __name__ == '__main__':
    txtList = [None for i in range(100)]

    conv_args = []
    for i in range(100):
        conv_args.append((i,))

    with Pool(8) as p:
        mean_intensities = p.map(giveText, conv_args)

    print(txtList)