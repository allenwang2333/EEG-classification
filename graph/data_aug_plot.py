import matplotlib.pyplot as plt

# Data for each model's test accuracy per subject
eegnet_accuracies = [0.7133, 0.6840, 0.6321,  0.6795]
multi_attention_accuracies = [0.6072, 0.5779, 0.5553, 0.5914]
eeglstmnet_accuracies = [0.6433, 0.6298, 0.6050, 0.6456]
eegattentionnet_accuracies = [0.6117, 0.6163, 0.4740, 0.5711]

# Subject numbers
subjects = list(range(len(eegnet_accuracies)))

# Plotting
plt.figure(figsize=(4, 7))
plt.plot(subjects, eegnet_accuracies, label='EEGNet', marker='o')
plt.plot(subjects, multi_attention_accuracies, label='MultiAttention', marker='o')
plt.plot(subjects, eeglstmnet_accuracies, label='EEGLSTMNet', marker='o')
#plt.plot(subjects, eegattentionnet_accuracies, label='EEGAttentionNet', marker='o')

# Custom x-axis labels
custom_labels = ['raw','guassian', 'channel_drop', 'guassian + channel_drop']
plt.xticks(subjects, custom_labels)  # Applying custom labels with rotation for readability

# Labeling
plt.xlabel('Data Augmentation')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Data Augmentation')
plt.legend()

# Show plot
plt.show()