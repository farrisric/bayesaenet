from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import glob

# Specify the path to your events file
events_files = glob.glob("/home/g15farris/bin/bayesaenet/bnn_aenet/logs/train_lrt_*/**/events.out.*", recursive=True)


for events_file in events_files:
    print(events_file)
    # Load the event accumulator
    event_acc = EventAccumulator(events_file)
    event_acc.Reload()

    # Access the available tags
    try:
        name = events_file.split('/')[7]
        # Get scalar values for a specific tag
        epochs = [i.value for i in event_acc.Scalars("epoch")]  # Replace "loss" with your tag
        rmse_train = [i.value for i in event_acc.Scalars("rmse/train")] 
        rmse_val = [i.value for i in event_acc.Scalars("rmse/val")] 

        plt.plot(epochs, rmse_train, label = f'val {name}')
        plt.plot(epochs, rmse_val, label = f'val {name}')
        plt.legend()
        plt.ylim(0, 10)
        plt.savefig('/home/g15farris/bin/bayesaenet/results/PdO/bnn_lrt/train/ciao.png')    
    except:
        print('bad events')
    # for epoch, train, val in zip(epochs, rmse_train, rmse_val):
    #     print(f"Epoch: {epoch.value}, rmse/train: {train.value}, rmse/val {val.value}")