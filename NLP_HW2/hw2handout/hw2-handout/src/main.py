import random
# train = open("C:/Users/chlee/PycharmProjects/NLP_HW1/hw01handout/ptb.2-21.txt",'r')
# train = open("C:/Users/chlee/PycharmProjects/NLP/NLP_HW2/hw2handout/hw2-handout/dev_text.txt", encoding='UTF8')
train = open("../dev_text.txt", encoding='UTF8')
train_lines = train.readlines()
# train_label = open("C:/Users/chlee/PycharmProjects/NLP_HW1/hw01handout/ptb.2-21.tgs",'r')
train_label = open("../dev_label.txt", encoding='UTF8')
train_labels = train_label.readlines()

print("train_lines[0]", train_lines[0].split('\n')[0])
print("len of train_lines", len(train_lines))
print("train_labels[0]", train_labels[0].split('\n')[0])
print("len of train_labels", len(train_labels))
#
# for i in range(10):
#     train_sample_file = open("C:/Users/chlee/PycharmProjects/NLP_HW1/hw01handout/ptb.2-21-" + str(i+1) + ".txt",'w')
#     train_label_sample_file = open("C:/Users/chlee/PycharmProjects/NLP_HW1/hw01handout/ptb.2-21-" + str(i + 1) + ".tgs", 'w')
#
#     train_sample = random.sample(train_lines, int(39832 / 10 * (i+1)))
#     for line in train_sample:
#
#         train_label_sample_file.write(train_labels[train_lines.index(line)])
#
#         train_sample_file.write(line)
#     print("finish")
#     train_sample_file.close()
#     train_label_sample_file.close()
# train.close()
#




import sys

def main():
    print("run main.py")
    # train_file = sys.argv[1]
    # test_file = sys.argv[2]


if __name__ == '__main__':
    main()
    print("main")

else :
    print("not main")
