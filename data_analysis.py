import os

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    file_list = os.listdir("results")
    results_dict = {}
    for file in file_list:
        str_list = file.split("_")
        alpha = str_list[2]
        beta = str_list[4]
        tau = str_list[6]
        batch_size = str_list[8]
        hidden = str_list[10]
        if alpha not in results_dict:
            results_dict[alpha] = {}
        if beta not in results_dict[alpha]:
            results_dict[alpha][beta] = {}
        if tau not in results_dict[alpha][beta]:
            results_dict[alpha][beta][tau] = {}
        if batch_size not in results_dict[alpha][beta][tau]:
            results_dict[alpha][beta][tau][batch_size] = {}
        if hidden not in results_dict[alpha][beta][tau][batch_size]:
            results_dict[alpha][beta][tau][batch_size][hidden]=[]

        score = []

        f = open("results/"+file,"r")
        for line in f:
            score.append(float(line))
        f.close()

        results_dict[alpha][beta][tau][batch_size][hidden].append(score)

    for alpha in results_dict:
        for beta in results_dict[alpha]:
            for tau in results_dict[alpha][beta]:
                for batch_size in results_dict[alpha][beta][tau]:
                    for hidden in results_dict[alpha][beta][tau][batch_size]:
                        plt.figure()
                        pre = 'Moving average of scores over 100 episodes\n'
                        title = 'alpha_' + str(alpha) + '_beta_' + \
                               str(beta) + '_tau_' + str(tau) + '_batch_' + str(batch_size) + \
                               '_hidden_' + str(hidden)
                        plt.title(pre+title)
                        plt.xlabel('Iteration')
                        plt.ylabel('Score')
                        for i in range(len(results_dict[alpha][beta][tau][batch_size][hidden])):
                            temp = results_dict[alpha][beta][tau][batch_size][hidden][i]
                            window = 100
                            average = []
                            for ind in range(len(temp) - window + 1):
                                average.append(np.mean(temp[ind:ind + window]))
                            plt.plot(average)
                        plt.ylim([-100,800])
                        plt.savefig('plots/'+title+'.png')
                        plt.show()


