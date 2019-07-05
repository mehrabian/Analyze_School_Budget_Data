df=pd.read_csv('')
# Print the summary statistics
print(df.describe())

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create the histogram
plt.hist(df['FTE'].dropna())
# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')

# Display the histogram
plt.show()
df.dtypes.value_counts()


LABELS=['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']
# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label,axis=0)

# Print the converted dtypes
print(df[LABELS].dtypes)

# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = df[LABELS].apply(pd.Series.nunique)

# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')



def compute_log_loss(predicted, actual, eps=1e-14):
     """ Computes the logarithmic loss between predicted and
     actual when these are 1D arrays.
     :param predicted: The predicted probabilities as floats between 0-1
     :param actual: The actual binary labels. Either 0 or 1.
     :param eps (optional): log(0) is inf, so we need to offset our
     predicted values slightly by eps from 0 or 1."""
     predicted = np.clip(predicted, eps, 1 - eps)
     loss = -1 * np.mean(actual * np.log(predicted)
     + (1 - actual)
      * np.log(1 - predicted))

     return loss
