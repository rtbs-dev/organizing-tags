---
title: Organizing Tagged Knowledge
subtitle: Similarity Measures and Semantic Fluency in Structure Mining
author:
  - Thurston Sexton
  - Mark Fuge
date: 8 August 2019
# use `echo "idetc19" | ./beamer`
---


# Background


## How to "Get Smart"?

Maintenance is expensive (\$50 billion in 2016) and expertise-driven, *but*...
**Smart manufacturing** technologies can reduce costs!

### SME's still not employing these technologies

- High Cost to implement; Risk is high with incorrect implementation
- Lack of Support/Expertise in manufacturing
- Leads to a lack of high quality sensor data

### Have no/little data
Difficult to assess impacts of new technologies

## How to "Get Smart"?



**Except**...*that's not entirely true.*

- Untapped source of data...natural language documents
- **Maintenance Work Orders**
- These come with severe issues! [@sexton2017hybrid]


::: {.alert title="jargon/misspellings"}
`Hyd leak at saw atachment`
:::

::: {.alert title="abbreviations"}
`HP coolant pressure at 75 psi`
:::

::: {.alert title='lack of context'}
`Replaced – Operator could
have done this!`
:::


## Want

What kinds of useable system knowledge could be gained from such data? Things that could assist in diagnostics, prognostics, schedule, etc.?

1. What **components**/concepts are **relevant** to our system

2. How these components are **related**

![](structuring.pdf)

## Faster Concepts through Tagging"

First step is not trivial! Previous work...

. . .

- Benefits to **tagging** [@sexton2017hybrid]

  - reduce annotation effort (\~ 50\%)
  - bottom-up concept definitions (adoption!)
  - collaborative annotations: "folksonomy" [@folksonomy-og]

. . .

- Bootstrap Human-Machine Interface
  - Present high importance tags for review [@nestor]
  - low-level organization
  - Less barrier to analysis [@sexton2018benchmarking]

## Tagging Example
\input{aux/mwo.tex}

. . .

Missing the second half: **tag relationships** (this work!)

$\rightarrow$ Structure Mining (e.g. Representation Learning)

::: notes
Basically by definition...tags have these benefits *because* they are unstructured input, requiring a lot less oversight/training/behavior-change.
:::


# Modeling Tag Relationships

## Overview

Two main ways relationship strength is modeled:

1. Context (global)
2. Sequence (local)

. . .

Each has specific strengths (and weaknesses) due to how they approach tag similarity

## Similar By Context

Across our entire set of tagged resources, what tags are correlated? Summarized as:

> When tags occur in a similar context, they are similar.

I.e. the **frequency** of tag **co-occurrences**.  Implies we see tags as **un-ordered**. (*See: Bag of Words*).

. . .

**Binary Indicator**: For each resource in a corpus $C$, a tag $t_k$ either "happens" or "doesn't happen":   $u_k= \{\mathbf{1}_c(t_k): c\in C\}$. This tag's "context vector" can be used to compare with other tags.

### Cosine Similarity
\begin{equation}\label{eq:cosine}
	s(t_1, t_2) =  \frac{u_1\cdot u_2}
	{\norm{u_1}\norm{u_2}}
\end{equation}

::: notes
So why do we care about Cosine? Can't we just take the norm between these vectors? This is where our preference for *context* comes in!
:::

## Cosine Similarity
Example:
\begin{align*}
	u_{\text{saw}} &= (1, 0, 0,\cdots)\\
	u_{\text{hydraulic}} &= (1, 1, 0, \cdots)\\
	u_{\text{operator}} &= (0, 0, 1, \cdots)
\end{align*}

### MWO 1: "Hyd leak at saw attachment"
\[ `hydraulic`, `saw` \] $

### MWO 2: "Hydraulic Leak at cutoff unit; Missing fitting replaced"
\[ `hydraulic`, `cutoff_unit`, `fitting` \]

### MWO 3: "Replaced – Operator could have done this!"
\[ `operator` \]


## Cosine Similarity
:::::::::::::: {.columns}
::: {.column width="40%"}

![](cosine.pdf)
:::
::: {.column width="60%"}

Geometric Example

::: {.alert title="Euclidean distance"}
Penalizes these models for having widely separated weights in each dimension. *This gets much worse in higher dimensions.*
:::
::: {.example title="Cosine similarity"}
Captures the intuition that relative context is key. *Robust to higher dimensions.*
:::
:::
::::::::::::::

## Cosine Similarity

Summary

- Picks up relationships between tags that co-occur, even under *widely varying contexts*. Global "structure" is quickly estimated.
- Highly scale-able, computationally
- Common in literature on tags, NLP, and discrete representation learning
	- Folksonomy $\rightarrow$ Taxonomy (Heymann Algorithm [@folksonomy-cosine])
	- Hyperbolic Embeddings for hierarchical learning (Facebook AI [@nickel2018learning])

::: {.example title="Pros"}
Excellent **recall**, easy to compute
:::

## Cosine Similarity

But what if tags are not mutually similar? What if:

-  `cutoff_unit` $\implies$ `fitting` but  
- `cutoff_unit` $\centernot{\implies}$`hydraulic` OR `fitting` $\centernot{\implies}$`hydraulic`

So, where/how do we draw the line?

- How do we threshold **signal** vs. **noise**?
- *a priori*?

::: {.alert title="Cons"}
difficult to tune, low/uncertain **precision**
:::

## Similar by Sequence

Tags come "out" **in order**!

. . .

It's potentially reasonable to assume ...

> The tag I remember first affects the one I recall next.

Each subsequent tag is conditional on one or more of its predecessors.

> Probability of observing a tag based on the previous, length-$n$ sequence tags via an $n$\ts{th}-order **Markov Chain**

Each "tag" is a **state** with transition probabilities to other tags. Tags on a resource are observed random walks through tag-states.
(See: $n$\ts{th}-order language model [@positional-language-model].)

## Markov Chain Random Walks

::::::::::::: {.columns}
::: {.column width=40%}

![](state.pdf)
:::

::: {.column width=60%}

::: {.example title="Random Walk 1"}
$\text{hydraulic} \rightarrow \text{cutoff\_unit} \rightarrow \text{fitting}$
:::

::: {.alert title="Random Walk 2"}
$\text{hydraulic} \rightarrow \text{saw}$
:::

### Random Walk 3
operator

:::
:::::::::::::


## Markov Chain

In an $n$\ts{th}-order tag Markov model, the probability of observing any $i$\ts{th} tag in a sequence is $P(t_i|t_{i-1},\cdots,t_{i-n})$.

To estimate the relationship strength of two tags (symmetric **edge probability**), as probability of observing them in a sequence:
\begin{equation}
	s(t_1, t_2) = \max \left[P(t_1|t_2), P(t_2|t_1)\right]
\end{equation}

## Markov Chain
Summary

- Strictly finds contemporaneous relationships. Local "structure" is prioritized.
- Common to model human generative processes as sequential
	- Hidden Markov Models
	- Recurrent Neural Networks
- Less likely to mistake co-occurrence for similarity

::: {.example title="Pros"}
Better **precision**
:::

::: {.alert title="Cons"}
Potential to miss **hidden patterns**; high computational cost
:::

## Compare

Let's step back again

1. Context (global) - Cosine Similarity
	- Excellent Recall;
	- Used in structure mining/representation learning literature
	- Hard to tune, mistakes correlation with relation
2. Sequence (local) - Markov Chain
	- Better Precision;
	- Used to model generative human processes
	- possibility to miss large-scale patterns

Which one? How to combine?

. . .

Let's step back to dive into what *exactly* might be happening.

# Semantic Fluency

## Audience Participation

**VOLUNTEER NEEDED**

. . .

What patterns were there? Why did *those* transitions happen?

## Another list :
Semantic (or verbal) Fluency Tests ask participants to recall objects in a category and write them down, as quickly as possible.

:::::::::::::: {.columns}
::: {.column width="30%"}

::: incremental
- dog
- cat
- lion
- tiger
- elephant
- wolf
- ...
:::
:::

. . .

::: {.column width="30%"}
### Hosehold

### Felids

### "African"

### Canids

:::
:::::::::::::

## What have we learned?

Key observations:

1. Recalled items **jump** between overlapping **contexts** (global)
2. **No** item **repetition** through the list (local)
3. **Past** items influence **sequential** context-changes

. . .

This describes a combination of *both* modelling archetypes! How to describe computationally? **Initial-Visit Emitting Random Walks**


# Initial-Visit Emitting Random Walks


## INVITE - Jun et al. (2015); Zemla \& Austerweil (2018) [@invite; @u-invite]

![](ivt_state.pdf)




::: {.example title="Random Walk 1"}
$\text{hydraulic} \rightarrow \text{cutoff\_unit} \rightarrow \text{fitting}$
:::

::: {.alert title="Random Walk 2"}
$\text{hydraulic} \rightarrow \text{saw}$
:::


## Example

![](walshetal/drivetrain_inviteRW2.pdf){ width=90% }

Component network model from Walsh et al. [@hannahs-nets]

## INVITE

:::::::::: {.columns}
::: {.column width=50%}
Since all visited nodes are "allowed" and *hidden*, there are an **infinite** number of paths possible to generate a given observation!

::: {.example title="key insight"}
Split each "random walk" of length $K$ into $K-1$ **absorbing** random sub-walks.
:::
:::
::: {.column width=50%}
![](absorb.pdf)
:::
:::::::::::
## INVITE - Absorbing Random (sub-) Walks

Partition at $k$\ts{th} *observed* transition ($t_k \rightarrow t_{k+1})$:

- $q$ transient states
- transition matrix $\mathbf{Q}^{(k)}_{q\times q}$
- $r$ absorbing states with $q\rightarrow r$ transitions as $\mathbf{R}^{(k)}_{q\times r}$

Markov transition matrix $\mathbf{M}^{(k)}_{n\times n}$ has the form:
\begin{equation}
	\mathbf{M}^{(k)} =
	\begin{pmatrix}
		\mathbf{Q}^{(k)}  & \mathbf{R}^{(k)} \\
		\mathbf{0}        & \mathbf{I}
	\end{pmatrix}
\end{equation}


. . .

Probability $P$ of chain starting at $t_k$ being *absorbed* into state $k+1$, letting $\mathbf{N} = \left( \mathbf{I}-\mathbf{Q} \right) ^{-1}$, is [@doyle2000random]:

\begin{equation}\label{eq:absrb}
	P\left(t_{k+1} \middle| t_{1:k},\mathbf{M}\right) =
		\left.\mathbf{N}^{(k)}R^{(k)}\right|_{q,1}
\end{equation}


## Summary

The probability of being absorbed at $k+1$ conditioned on jumps $1:k$ is the probability of observing our $k+1$ INVITE tag.


## Model Inference

Likelihood of $\mathbf{M}$ given observed *censored* chain $\vec{t}$ is:

\begin{equation}
	\Like\left(\vec{t} \,|\, \theta\,; \mathbf{M}\,\right) =
		\theta(t_1)\prod_{k=1}^{T-1} P\left(t_{k+1}\,\middle|\ t_{1:k} \,;\,\mathbf{M}\,\right)
\end{equation}

This implies that a "folksonomy" of tag lists $\mathbf{C} = \left\{ \vec{t}_1, \vec{t}_2, \cdots, \vec{t}_{c} \right\}$ can recover an $\mathbf{M}$ through optimization:

\begin{equation}\label{eq:opt}
	\mathbf{M}^* \leftarrow \argmin_{\mathbf{M}} \quad
	\sum_{i=1}^{C}
	\sum_{k=1}^{T_i-1}
		-\log \Like \left(t^{(i)}_{k+1} \middle| t^{(i)}_{1:k},\mathbf{M}\right)
\end{equation}

## Model Inference - Implementation Details

How to optimize?

- Still nearly intractable for large numbers of tags, given search-space
- Analytic gradient given in Jun \& Zemla et al. [@invite] has restrictions
- Binarizing edge states [@u-invite] removes edge weights entirely.

. . .

We leverage **automatic differentiation**

- Tensor library PyTorch to minimize the loss via ADAM [@paszke2017automatic]
- Flexible w.r.t symmetry, bound transformations, etc during training


# Exp. 1 - Synthetic Example

## Experiment 1 - Setup

What are we missing if a process like INVITE *is* generating our data?

- If we assume Bag-of-Words applies
- If we assume Markov property applies

. . .

**Let's Find Out** - We are looking to quantify model effects on *edge probability*.

- **recall** is our ability to select all relevant edges, and
- **precision** is the relevance of our selected edges.

The edge probability threshold ($\sigma$) we choose has a huge effect on these.



## Experiment 1 - Example

To illustrate, return to the example of Walsh et al.  [@hannahs-nets]

- adjacency matrix representation (recovered edge probabilities):
- $C=20$ random walks of length $l=4$

![](walshetal/matrix_compare_modeldrivetrain_nwalks20_length4_seed2_saveTrue.pdf)

. . .

Let's threshold for optimal $F_1$ score...

------------
\ \

![](walshetal/sensitivitynet_INVITE_modeldrivetrain_nwalks20_length4_seed2_saveTrue.pdf){width=50%}\ ![](walshetal/sensitivitynet_Cosine_modeldrivetrain_nwalks20_length4_seed2_saveTrue.pdf){width=50%}


 ![](walshetal/sensitivitynet_MC1_modeldrivetrain_nwalks20_length4_seed2_saveTrue.pdf){width=50%}\ ![](walshetal/sensitivitynet_MC2_modeldrivetrain_nwalks20_length4_seed2_saveTrue.pdf){width=50%}


## Synthetic Examples
Need to test model behavior under varying

- graph sizes
- resource quantity (\# random walks)

We randomly generate Watts-Strogatz graphs (N=90) with varying numbers of "tags" and corpus-sizes.

Compare the precision/recall of each similarity measure archetype.



## Experiment 1 - Summary Curves

Across 90 graphs, $N,C\in\{10,25,50\}$:

![](randomgraphs/p_v_r_graphs90_nodes10-25-50_nwalks10-25-50_length4_seed8_readinTrue_saveTrue.pdf){width=50%}\ ![](randomgraphs/t_v_r_graphs90_nodes10-25-50_nwalks10-25-50_length4_seed8_readinTrue_saveTrue.pdf){width=50%}

- **Precision**: INVITE ~ Sequential Probabilities
- **Recall**: INVITE ~ Contextual Similarity


## Experiment 1 - Performance Curves
\ \

![](randomwalks_fscoresA.pdf)

## Experiment 1 - Performance Curves
\ \

![](randomwalks_fscoresB.pdf)

## Experiment 1 - Performance Curves
\ \

![](randomwalks_fscoresC.pdf)



# Exp. 2 - Real-World Case Study



## Excavator Dataset

1. **Mining Dataset**: Excavator Maintenance Work orders [@excavator-dataset; @excavator-data]
	- 8 Excavators, 8264 MWOs
	- Bespoke keyword-recognition tool; recognize failure "major subsystem"
	- Labor intensive... months (and a dissertation!)

2. Tags - Nestor Toolkit [@nestor]
	- Compare Survival Analysis - tags vs. keyword-recognition [@sexton2018benchmarking]
	- Subsystems approximated by expert-determined "tag-sets"

 . . .

### PROBLEM
how to estimate subsystem by tags? **Which tags**?

## Excavator Data

Some tags are trivially known

- `bucket` $\rightarrow$ "Bucket Subsystem"
- `hydraulic` $\rightarrow$ "Hydraulic System"
- `engine` $\rightarrow$ "Engine Subsystem"

Thanks to the keyword-tagger, we *also* have ground-truth tag-subsystem allocations:

- `hose` : *probably* **hydraulic**
- `teeth`: *always* **bucket**
- `bolt`: all of them?


## Excavator Ground Truth
*All tags:*

![](excavators/tag_multinomial_ntags3_freq5_topn50_thres60_saveTrue.pdf){width=90%}

Color \approx classification into subsystem (\geq 60\% of occurrences).

## Semi-supervised Learning
More generally, this is a problem of estimating labels from the data *topology*; **Label Spreading**

![](local-global-consistency.pdf)

We apply the label spreading algorithm of Zhou et al (2004) [@zhou2004learning]

- Only using the three trivially-labeled tags
- MWOs with at least 3 tags, each occurring at least 10x
	- $C=263$ MWOs, $N=40$ tags
- Comparing ground-truth distributions to classification score

## Excavator - Results

![](excavators/F1_KL_ntags3_freq5_topn50_thres60_saveTrue.pdf)

## Excavator - Distribution Comparison

at optimal $F_1$-score:

![](excavators/ternary_ntags3_freq5_topn50_thres60_saveTrue.pdf)

## Excavator - Recovered Structure

:::::::::::: {.columns}
::: {.column width=50%}
![](excavators/MC1_network_ntags3_freq5_topn50_thres60_saveTrue.pdf){ height=90%}
:::
::: {.column width=50%}
![](excavators/MC2_network_ntags3_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::::::::::::

## Excavator - Recovered Structure

:::::::::::: {.columns}
::: {.column width=50%}
![](excavators/INVITE_network_ntags3_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::: {.column width=50%}
![](excavators/Cosine_network_ntags3_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::::::::::::

# Conclusions

## Key Contributions

1. If censored random walks *are* taking place

	- Context-based models could work; **hard to tune**
	- Sequential models can **miss latent** relationships
	- Accounting for censoring improves **precision and recall**

. . .

2. In analytics tasks down-stream (i.e. Semi-Supervised Classifier)

	- Use similarity model assumptions as pre-processor
	- Incorporating INVITE can better map to users' organizational intuitions.

## Future Work

INVITE is only the beginning...

- Relevant tags may be skipped entirely

	- Too general
	- Too specific
	- Hidden nodes? Node hierarchies?

. . .

- Active Learning of representation learning

	- Real-time feedback to *build trust*
	- Exploit embeddings, probabilistic models, etc.
	- Mixture models for different *types* of relationships

# Thank You! Questions?

# Backup

## Animals Network - Goni et al. (2011) [@goni2011semantic]

![](animals.png)

## Excavator - Results
- MWOs with at least 2 tags, each occurring at least 10x
  - $C=1712$

![](excavators/F1_KL_ntags2_freq5_topn50_thres60_saveTrue.pdf)

## Excavator - Distribution Comparison

at optimal $F_1$-score:
![](excavators/ternary_ntags2_freq5_topn50_thres60_saveTrue.pdf)

## Excavator - Recovered Structure

:::::::::::: {.columns}
::: {.column width=50%}
![](excavators/MC1_network_ntags2_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::: {.column width=50%}
![](excavators/MC2_network_ntags2_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::::::::::::

## Excavator - Recovered Structure

:::::::::::: {.columns}
::: {.column width=50%}
![](excavators/INVITE_network_ntags2_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::: {.column width=50%}
![](excavators/Cosine_network_ntags2_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::::::::::::

# References

## References {.allowframebreaks}
