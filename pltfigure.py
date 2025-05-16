import matplotlib.pyplot as plt  # Importing the matplotlib library for plotting

# Assuming the previous code has been executed and y_test_actual and y_pred are defined

# Visualization part
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='Actual Traffic')
plt.plot(y_pred, label='Predicted Traffic')
plt.title('Traffic Volume Prediction')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.legend()
plt.show()
