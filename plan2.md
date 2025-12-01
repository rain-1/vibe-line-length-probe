# logistic probes

we want to new attach 70 probes to the transformer (where?)
we will be training probe[i]=1 and every other probe=0 when our data set says line length at this point in the text is 'i'
we can apply one-hot to our data set, to get vectors for training our probes
if the line length is >=70 we can just drop that row from our dataset. because there's not much signal in training the probes on all zeros.

we should do a proper wandb logged training run with a validation data set.
lets start small and scale up to the full data set once it works.
