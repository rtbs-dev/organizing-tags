---
title: Organizing Tagged Knowledge
subtitle: Similarity Measures and Semantic Fluency in Structure Mining
author:
  - Thurston Sexton (NIST)
  - Mark Fuge (UMD)
date: 8 August 2019
# use `echo "idetc19" | ./beamer`
---



# Background


## How to "Get Smart"?

Maintenance is expensive (\$50 billion for USA in 2016 [@thomas2018costs]) and expertise-driven, *but*...
**Smart manufacturing** technologies can reduce costs!

### SME's still not employing these technologies

- High Cost to implement; Risk is high with incorrect implementation
- Lack of Support/Expertise in manufacturing
- Leads to a lack of high quality (or understood) sensor data

### Have little/no data
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


## Tagging Example

\input{aux/mwo.tex}

. . .

Missing the second half: **tag relationships** (this work!)

$\rightarrow$ Structure Mining (e.g. Representation Learning)

::: notes
Basically by definition...tags have these benefits *because* they are unstructured input, requiring a lot less oversight/training/behavior-change.
:::


# Modeling Tag Relationships

## Compare the Archetypes

At a high level:

1. Context (global) - *e.g. Cosine Similarity*
    
    > Summary: "Tags found in similar contexts *are similar*."
	- Excellent Recall;
	- Used in structure mining/representation learning literature
	- Hard to tune, mistakes correlation with relation

. . .

2. Sequence (local) - *e.g. Markov Chain*
    
    > Summary: Tags come "out" of the user *in order*!
	- Better Precision;
	- Used to model generative human processes
	- possibility to miss large-scale patterns

Which one? How to combine?


## Cosine Similarity

:::::::::::::: {.columns}
::: {.column width=60%}

Cosine of the angle between example vectors:

### MWO 1: "Hyd leak at saw attachment"
\[ `hydraulic`, `saw` \] $

### MWO 2: "Hydraulic Leak at cutoff unit; Missing fitting replaced"
\[ `hydraulic`, `cutoff_unit`, `fitting` \]

### MWO 3: "Replaced – Operator could have done this!"
\[ `operator` \]

:::
::: { .column width=40%}


\begin{align*}
	u_{\text{saw}} &= (1, 0, 0,\cdots)\\
	u_{\text{hydraulic}} &= (1, 1, 0, \cdots)\\
	u_{\text{operator}} &= (0, 0, 1, \cdots)
\end{align*}

![](cos.pdf)
:::
:::::::::::::::::

## Markov Chain

::::::::::::: {.columns}
::: {.column width=40%}

![](state.pdf)
:::

::: {.column width=60%}

Each "tag" is a **state** with transition probabilities to other tags. Tags on a resource are observed random walks through tag-states.

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



## Hold up a minute...

Let's step back. What might actually be going on? What could a reasonable set of tags be coming from?

::: {.example title="Random Walk 1"}
$\text{hydraulic} \rightarrow \text{cutoff\_unit} \rightarrow \text{fitting}$
:::

::: {.alert title="Random Walk 2"}
$\text{hydraulic} \rightarrow \text{saw}$
:::

![](ivt_state.pdf){ width=70% }


## More Realistic Example

![](walshetal/drivetrain_inviteRW2.pdf){ width=90% }

Component network model from Walsh et al. [@hannahs-nets]




# Semantic Fluency Data

## Audience Participation

**VOLUNTEER NEEDED**

. . .

What patterns were there? Why did *those* transitions happen?

## Another list :
Semantic (or verbal) Fluency Tasks ask participants to recall objects in a category and write them down, as quickly as possible.

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
### Household

### Felines

### "African"

### Canines

:::
:::::::::::::

## What have we learned?

Key observations:

1. Recalled items **jump** between overlapping **contexts** (global)
2. **No** item **repetition** through the list (local)
3. **Past** items influence **sequential** context-changes

. . .

This describes a combination of *both* modelling archetypes! How to describe computationally? **Initial-Visit Emitting Random Walks**


## INVITE - Jun et al. (2015); Zemla \& Austerweil (2018) [@invite; @u-invite]

:::::::::: {.columns}
::: {.column width=50%}

- All visited nodes are "allowed" and *hidden*
- **infinite** number of paths can generate a given observation!

::: {.example title="key insight"}
Split each "random walk" of length $K$ into $K-1$ **absorbing** random sub-walks.
:::
:::
::: {.column width=50%}
![](absorb.pdf)
:::
:::::::::::




# Experiments

## Research Question

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



# Exp. 2 - Excavator Case Study



## Excavator - Dataset

1. **Mining Dataset**: Excavator Maintenance Work orders [@excavator-dataset; @excavator-data]
	- 8 Excavators, 8264 MWOs
	- Bespoke keyword-recognition tool; recognize failure "major subsystem"
	- Labor intensive... months (and a dissertation!)

2. Tags - Nestor Toolkit [@nestor]
	- Compare Survival Analysis - tags vs. keyword-recognition [@sexton2018benchmarking]
	- Subsystems approximated by expert-determined "tag-sets"

### PROBLEM
How to estimate subsystem by tags? **Which tags**?


## Excavator - Recovered Structure

:::::::::::: {.columns}
::: {.column width=50%}
![](excavators/INVITE_network_ntags3_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::: {.column width=50%}
![](excavators/Cosine_network_ntags3_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::::::::::::

## Excavator - Recovered Structure

:::::::::::: {.columns}
::: {.column width=50%}
![](excavators/MC1_network_ntags3_freq5_topn50_thres60_saveTrue.pdf){ height=90%}
:::
::: {.column width=50%}
![](excavators/MC2_network_ntags3_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::::::::::::


## Excavator Data

Some tags are trivially known (use these!)

- `bucket` $\rightarrow$ "Bucket Subsystem"
- `hydraulic` $\rightarrow$ "Hydraulic System"
- `engine` $\rightarrow$ "Engine Subsystem"


Thanks to the keyword-tagger, we can compare ground-truth tag-subsystem allocations:

- `hose` : *probably* **hydraulic**
- `teeth`: *always* **bucket**
- `bolt`: all of them?



## Semi-supervised Learning

More generally, this is a problem of estimating labels from the data *topology*; **Label Spreading**

![](local-global-consistency.pdf)

We apply the label spreading algorithm of Zhou et al (2004) [@zhou2004learning]



## Excavator - Results
Comparing tag label distributions: **ground-truth** (from keyword extractor) v.s. **predicted** classification (from label spreading)

. . .

\begin{table}
\centering
    \begin{tabular}{lrrrr}
    \toprule
    {} &   $F_1^*$ &  $\Sigma KL$ &  $\mu_{KL}$ \\
    \midrule
    INVITE &  {\bf 0.83} &    {\bf 13.5} &    {\bf 0.35} \\
    Cosine &  0.71 &    16.5 &    0.42 \\
    MC1    &  0.80 &    17.0 &    0.44 \\
    MC2    &  0.66 &    17.6 &    0.45 \\
    \bottomrule
    \end{tabular}
\end{table}


### INVITE consistently performs best, across a wide range of $\sigma$.



# Conclusions

## Key Contributions - Modeling Similarity

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



## Excavator - Results ($l\geq3$)

![](excavators/F1_KL_ntags3_freq5_topn50_thres60_saveTrue.pdf)

## Excavator - Distribution Comparison ($l\geq3$)

at optimal $F_1$-score:

![](excavators/ternary_ntags3_freq5_topn50_thres60_saveTrue.pdf)

## Excavator - Results ($l\geq2$)
- MWOs with at least 2 tags, each occurring at least 10x
  - $C=1712$

![](excavators/F1_KL_ntags2_freq5_topn50_thres60_saveTrue.pdf)

## Excavator - Distribution Comparison ($l\geq2$)

at optimal $F_1$-score:
![](excavators/ternary_ntags2_freq5_topn50_thres60_saveTrue.pdf)

## Excavator - Recovered Structure ($l\geq2$)

:::::::::::: {.columns}
::: {.column width=50%}
![](excavators/MC1_network_ntags2_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::: {.column width=50%}
![](excavators/MC2_network_ntags2_freq5_topn50_thres60_saveTrue.pdf){ height=90% }
:::
::::::::::::

## Excavator - Recovered Structure ($l\geq2$)

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
