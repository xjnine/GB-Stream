# **MGB-Stream**

Taking inspiration from multi-granularity cognitive learning, we have designed a novel data streaming clustering method utilizing the granular-ball computing approach, which is
fully adaptive and capable of handling evolving data streams.

## Algorithmic Processes

------

![MGB-Steam](/notebooks/images/i.png)

MGB-Stream is constructed on four core components: the Initialization Module (IM) is responsible for generating initial 𝑀𝐺𝐵𝑠; the Update Module (UM) handles the continuous updating of 𝑀𝐺𝐵𝑠; the Update and Elimination Module (EM) is tasked with the updating of the weights and the elimination of the old historical data; finally, the Clustering Module (CM) is responsible for integrating 𝑀𝐺𝐵𝑠 to form the final clusters when a clustering request is received. In the entire life cycle of the algorithm, it is divided into two vertical phases: initialization and maintenance. Meanwhile, in terms of operational flow, it is distinguished into online and offline phases: the online phase involves the updating of 𝑀𝐺𝐵𝑠 by the UM component, as well as the adjustment of the weights and the elimination of the historical data by the EM component; while in the offline phase, the CM component handles the clustering requests in order to generate the macro clusters.

## Files

------

These program mainly containing:

| folder/document | **explanation**                                              |
| --------------- | ------------------------------------------------------------ |
| **data**        | The folder where the dataset is stored.                      |
| **models**      | Algorithm code file.                                         |
| **main**        | The algorithm starts from here.                              |
| **notebooks**   | Store files associated with markdown, as well as the output files of the algorithm. |

## Requirements

------

For requirements on dependent packages, see the requirements.txt file.

### Dataset Format

The format is csv, the last column of data is the timestamp and the penultimate column is the label.

## Usage

------

The current sample dataset is RBF3 from the paper. There are 5 flow rates configured in main.py and can be run directly.

