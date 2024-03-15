import matplotlib.pyplot as plt

# Data for each model's test accuracy per subject
eegnet_accuracies = [0.6185, 0.6749, 0.7133,  0.6659,  0.6140]
multi_attention_accuracies = [0.5598, 0.5914, 0.6072, 0.6117, 0.5576]
eeglstmnet_accuracies = [0.5779, 0.6343, 0.6433, 0.6230, 0.6208]
eegattentionnet_accuracies = [0.6230, 0.5982, 0.6117, 0.5102, 0.3815]

# Subject numbers
subjects = list(range(len(eegnet_accuracies)))

# Plotting
plt.figure(figsize=(8,6))
plt.plot(subjects, eegnet_accuracies, label='EEGNet', marker='o')
plt.plot(subjects, multi_attention_accuracies, label='EEGMultiAttentionNet', marker='o')
plt.plot(subjects, eeglstmnet_accuracies, label='EEGLSTMNet', marker='o')
#plt.plot(subjects, eegattentionnet_accuracies, label='EEGAttentionNet', marker='o')

# Custom x-axis labels
custom_labels = ['time 200', 'time 400', 'time 600', 'time 800', 'time 1000']
plt.xticks(subjects, custom_labels)  # Applying custom labels with rotation for readability

# Labeling
plt.xlabel('Time bins', fontsize='large', fontweight='bold')
plt.ylabel('Test accuracy', fontsize='large', fontweight='bold')
plt.title('Test accuracy per model per time bin', fontsize='x-large', fontweight='bold')
plt.legend(fontsize='medium', title_fontsize='large')

# Show plot
plt.show()