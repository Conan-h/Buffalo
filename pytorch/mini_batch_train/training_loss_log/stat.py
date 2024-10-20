
import statistics
import os
import matplotlib.pyplot as plt

def draw(y, string):
    plt.clf()
    ll = len(y)
    x = list(range(1, ll+1))  # Generates a list from 1 to 200

    # Create the line plot
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel('# epoch')
    plt.ylabel(string)
    plt.title('Line Plot Example')

    # Save the plot to a PNG file
    plt.savefig(str(string) +'_line_plot.png')
    
    # Show the plot
    plt.show()


def read_files_in_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    dataset = ''
    for file_name in file_list:
        if not 'log' in file_name:
            continue
        if not 'train_loss' in file_name:
            continue
        # if not 'cora' in file_name:
        #     continue
        # dataset = 'cora'
        # if not 'arxiv' in file_name:
        #     continue
        # dataset = 'arxiv'
        if not 'pubmed' in file_name:
            continue
        dataset = 'pubmed'
        file_path = os.path.join(folder_path, file_name)
        print('file------ ', file_name)
        loss_list=[]
        
        if os.path.isfile(file_path) :  # Ensure it is a file, not a subfolder
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("training loss"):
                        # print(line.split(" "))
                        loss = float(line.split(" ")[3].strip())
                        
                        loss_list.append(loss)
                        
        
        # print('f1_list ', f1_list)
        draw(loss_list, str(dataset) + ' training loss ')
        # print()
    print('max loss_list ',max(loss_list))
    print('min loss_list ',min(loss_list))
    return loss_list



numbers = read_files_in_folder('./')
length = len(numbers)
start = 300
end = 360
mean = statistics.mean(numbers[start: end])
std_dev = statistics.stdev(numbers[start: end])
formatted_result = f"{mean:.8f} ± {std_dev:.8f}"
print(formatted_result)