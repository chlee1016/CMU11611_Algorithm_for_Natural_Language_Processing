import random
train = open("C:/Users/chlee/PycharmProjects/NLP_HW1/hw01handout/ptb.2-21.txt",'r')
train_lines = train.readlines()
train_label = open("C:/Users/chlee/PycharmProjects/NLP_HW1/hw01handout/ptb.2-21.tgs",'r')
train_labels = train_label.readlines()

for i in range(10):
    train_sample_file = open("C:/Users/chlee/PycharmProjects/NLP_HW1/hw01handout/ptb.2-21-" + str(i+1) + ".txt",'w')
    train_label_sample_file = open("C:/Users/chlee/PycharmProjects/NLP_HW1/hw01handout/ptb.2-21-" + str(i + 1) + ".tgs", 'w')

    train_sample = random.sample(train_lines, int(39832 / 10 * (i+1)))
    for line in train_sample:

        train_label_sample_file.write(train_labels[train_lines.index(line)])

        train_sample_file.write(line)
    print("finish")
    train_sample_file.close()
    train_label_sample_file.close()
train.close()


#################################
a = ['a','b','c','d','e','f','g','h','i','j']
# random.seed(1)
print(random.sample(a, 2))
print(a.index('b'))
# print(random.sample(a, 2))
# print(random.sample(a, 2))