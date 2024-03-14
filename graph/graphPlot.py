import matplotlib.pyplot as plt

# Data for each model's test accuracy per subject
eegnet_accuracies = [0.64, 0.52, 0.8, 0.56, 0.7659574468085106, 0.3877551020408163, 0.74, 0.72, 0.7659574468085106, 0.70, 0.4221]
multi_attention_accuracies = [0.58, 0.32, 0.76, 0.5, 0.5531914893617021, 0.3469387755102041, 0.6, 0.62, 0.7021276595744681, 0.54, 0.3995]
eeglstmnet_accuracies = [0.52, 0.28, 0.66, 0.58, 0.5531914893617021, 0.3877551020408163, 0.68, 0.7, 0.6170212765957447, 0.56, 0.3657]
eegattentionnet_accuracies = [0.62, 0.36, 0.66, 0.4, 0.5106382978723404, 0.3673469387755102, 0.64, 0.54, 0.6808510638297872, 0.64, 0.3409]

# Subject numbers
subjects = list(range(len(eegnet_accuracies)))

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(subjects, eegnet_accuracies, label='EEGNet', marker='o')
plt.plot(subjects, multi_attention_accuracies, label='MultiAttention', marker='o')
plt.plot(subjects, eeglstmnet_accuracies, label='EEGLSTMNet', marker='o')
plt.plot(subjects, eegattentionnet_accuracies, label='EEGAttentionNet', marker='o')

# Custom x-axis labels
custom_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'train on all\ntest on 0', 'train on 0\ntest on all']
plt.xticks(subjects, custom_labels)  # Applying custom labels with rotation for readability

# Labeling
plt.title('Test Accuracies vs Subject Number', fontsize='large', fontweight='bold')
plt.xlabel('Subject Number', fontsize='large', fontweight='bold')
plt.ylabel('Test Accuracy', fontsize='x-large', fontweight='bold')
plt.xticks(subjects)
plt.legend(fontsize='medium', title_fontsize='large')
plt.grid(True)

# Show plot
plt.show()
