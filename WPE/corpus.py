import numpy as np

from utils import listdir_nohidden, listfiles_nohidden


def experiment_files_AC_dirty():
    base = '<base location of the folder>'
    people = np.random.choice(listdir_nohidden(base), 2)

    mic_1 = np.random.choice(listdir_nohidden(people[0] + '/dirty/'), 1)
    mic_2 = np.random.choice(listdir_nohidden(people[1] + '/dirty/'), 1)

    return mic_1[0], mic_2[0]


def experiment_files_AC_no_noise():
    base = '<base location of the folder>'
    people = np.random.choice(listdir_nohidden(base), 2)

    mic_1 = np.random.choice(listdir_nohidden(people[0] + '/no_noise/'), 1)
    mic_2 = np.random.choice(listdir_nohidden(people[1] + '/no_noise/'), 1)

    return mic_1[0], mic_2[0]

def experiment_files_timit_train():

    base = '<base location of the folder>/TRAIN/'

    folders = listdir_nohidden(base)
    # print(folders)
    folders_1stlvl = np.random.choice(folders, 1)
    folders = listdir_nohidden(folders_1stlvl[0])
    folders_2ndlvl = np.random.choice(folders, 1)
    files = listfiles_nohidden(folders_2ndlvl[0])
    files = np.random.choice(files, 2)

    return files[0], files[1]

def experiment_files_timit_test():

    base = '<base location of the folder>/TEST/'

    folders = listdir_nohidden(base)
    # print(folders)
    folders_1stlvl = np.random.choice(folders, 1)
    folders = listdir_nohidden(folders_1stlvl[0])
    folders_2ndlvl = np.random.choice(folders, 1)
    files = listfiles_nohidden(folders_2ndlvl[0])
    files = np.random.choice(files, 2)

    return files[0], files[1]

def experiment_files_voc():
    base = '<base location of the folder>/data/'
    files = listfiles_nohidden(base)
    files = np.random.choice(files, 2)
    
    return files[0], files[1]

def experiment_files_voc_MF():
    base = '<base location of the folder>/data/'
    female = np.concatenate((np.arange(1,54), np.arange(72,202), np.arange(297,382), np.arange(460,554), np.arange(583,615), np.arange(655,709), np.arange(716,748), np.arange(816,900), np.arange(967,988), np.arange(1027,1067), np.arange(1225,1274), np.arange(1316,1344), np.arange(1368,1393), np.arange(1448,1466), np.arange(1472,1503), np.arange(1517,1525), np.arange(1532,1613), np.arange(1639,1689), np.arange(1712,1819), np.arange(2045,2060), np.arange(2095,2106), np.arange(2185,2225), np.arange(2250,2260), np.arange(2329,2453), np.arange(2545,2574), np.arange(2666,2676), np.arange(2717,2763)))
    male = np.concatenate((np.arange(54,72), np.arange(202,297), np.arange(382,460), np.arange(554,583), np.arange(615,655), np.arange(709,716), np.arange(748,816), np.arange(900,967), np.arange(988,1027), np.arange(1067, 1225), np.arange(1274, 1316), np.arange(1344, 1368), np.arange(1393, 1448), np.arange(1466, 1472), np.arange(1503, 1517), np.arange(1525, 1532), np.arange(1613, 1639), np.arange(1689, 1712), np.arange(1819, 2045), np.arange(2060, 2095), np.arange(2106, 2185), np.arange(2225, 2250), np.arange(2260, 2329), np.arange(2453, 2545), np.arange(2574, 2666), np.arange(2676, 2717)))

    male_file = np.random.choice(male, 2)
    female_file = np.random.choice(female, 2)

    return base + create_filename(male_file[0]), base + create_filename(female_file[0])


def experiment_files_MTS():
    # base = '/Volumes/Promise_Pegasus/Teun_Corpora/vocalizationcorpus/data/'

    # base = '/home/visionlab/Teun/MapTask_Scots/'
    base = '<base location of the folder>/MapTask_Scots/'
    files = listdir_nohidden(base)

    c_files = np.random.choice(files, 2)

    return c_files[0], c_files[1]

def create_filename(number):
    if number >= 1000:
        return 'S' + str(number) + '.wav'
    elif number >= 100:
        return 'S0' + str(number) + '.wav'
    elif number >= 10:
        return 'S00' + str(number) + '.wav'
    else:
        return 'S000' + str(number) + '.wav'
