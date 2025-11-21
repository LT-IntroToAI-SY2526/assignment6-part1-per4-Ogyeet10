# Assignment 6 Part 1 - Writeup

**Name:** Aidan  
**Date:** 11/21/2025

---

## Part 1: Understanding Your Model

### Question 1: R² Score Interpretation

What does the R² score tell you about your model? What does it mean if R² is close to 1? What if it's close to 0?

**YOUR ANSWER:**
The R² score tells us how well the line fits the data. If it's close to 1, it means the model's predictions are very accurate and explain most of the data. If it's close to 0, it means the model isn't doing a good job and doesn't really explain the relationship between the variables.

---

### Question 2: Mean Squared Error (MSE)

What does the MSE (Mean Squared Error) mean in plain English? Why do you think we square the errors instead of just taking the average of the errors?

**YOUR ANSWER:**
MSE is just the average of the squared differences between what we predicted and the actual values. We square the errors so that positive and negative mistakes don't cancel each other out. It also makes larger errors stand out more, so we can see if any predictions were really far off.

---

### Question 3: Model Reliability

Would you trust this model to predict a score for a student who studied 10 hours? Why or why not? Consider:

- What's the maximum hours in your dataset?
- What happens when you make predictions outside the range of your training data?

**YOUR ANSWER:**
I'd trust it for 10 hours since the maximum in our dataset is 9.6 hours, which is pretty close. However, if we tried to predict for something way outside the range, like 20 hours, it probably wouldn't work well. The model might predict a score over 100%, which is only possible with extra credit.

---

## Part 2: Data Analysis

### Question 4: Relationship Description

Looking at your scatter plot, describe the relationship between hours studied and test scores. Is it:

- Strong or weak?
- Linear or non-linear?
- Positive or negative?

**YOUR ANSWER:**
The relationship is strong, positive, and linear. You can clearly see that as the hours studied go up, the test scores go up as well in a fairly straight line.

---

### Question 5: Real-World Limitations

What are some real-world factors that could affect test scores that this model doesn't account for? List at least 3 factors.

**YOUR ANSWER:**

1. Natural ability or prior knowledge of the subject.
2. How much sleep the student got before the test.
3. If the student was distracted or anxious during the exam.

---

## Part 3: Code Reflection

### Question 6: Train/Test Split

Why do we split our data into training and testing sets? What would happen if we trained and tested on the same data?

**YOUR ANSWER:**
We split the data to verify that the model can actually generalize to new information, not just memorize the answers. If we trained and tested on the same data, we would get a really high score, but we wouldn't know if the model would work correctly on real-world data it hasn't seen before.

---

### Question 7: Most Challenging Part

What was the most challenging part of this assignment for you? How did you overcome it (or what help do you still need)?

**YOUR ANSWER:**
The hardest part was probably getting the graphs to look exactly right with the labels and colors. I just looked at the example code provided and referenced the documentation to figure out the correct settings.

---

## Part 4: Extending Your Learning

### Question 8: Future Applications

Describe one real-world problem you could solve with linear regression. What would be your:

- **Feature (X):**
- **Target (Y):**
- **Why this relationship might be linear:**

**YOUR ANSWER:**
I could use it to predict house prices.

- **Feature (X):** Square footage of the house.
- **Target (Y):** Selling price.
- **Why this relationship might be linear:** Generally, as houses get bigger, the price goes up in a pretty consistent way.

---

## Grading Checklist (for your reference)

Before submitting, make sure you have:

- [x] Completed all functions in `a6_part1.py`
- [x] Generated and saved `scatter_plot.png`
- [x] Generated and saved `predictions_plot.png`
- [x] Answered all questions in this writeup with thoughtful responses
- [x] Pushed all files to GitHub (code, plots, and this writeup)

---

## Optional: Extra Credit (+2 points)

If you want to challenge yourself, modify your code to:

1. Try different train/test split ratios (60/40, 70/30, 90/10)
2. Record the R² score for each split
3. Explain below which split ratio worked best and why you think that is

**YOUR ANSWER:**
