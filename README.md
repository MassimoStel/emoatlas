<p align="center">
  <img src="EmoAtlas Logo.png" data-canonical-src="ea.png" width="200" height="200" />
</p>

# EmoAtlas

EmoAtlas is a Python package for cognitive-emotional text analysis. It transforms text into textual forma mentis networks and uses multilingual emotion lexicons to profile the emotional content of words, semantic frames, and full documents.

The package is designed for researchers, students, and analysts who want to move beyond simple sentiment polarity. EmoAtlas helps answer questions such as:

- Which concepts are central in a text?
- Which words frame a target concept such as `failure`, `trust`, `future`, or `paura`?
- Which emotions are over- or under-represented compared with an internal baseline?
- How do syntactic, synonym, and hypernym relations contribute to the same conceptual network?
- How can a network be exported for inspection, reporting, or downstream analysis?

Repository: https://github.com/MassimoStel/emoatlas

Matching Colab notebook prepared with this README: `EmoAtlas_full_tutorial_Colab.ipynb`. After committing the notebook to the repository root, this URL will open it directly in Google Colab:

```text
https://colab.research.google.com/github/MassimoStel/emoatlas/blob/main/EmoAtlas_full_tutorial_Colab.ipynb
```


Main reference: Semeraro, A., Vilella, S., Improta, R. et al. (2025). *EmoAtlas: An emotional network analyzer of texts that merges psychological lexicons, artificial intelligence, and network science*. Behavior Research Methods, 57, 77. https://doi.org/10.3758/s13428-024-02553-7

---

## Contents

1. What EmoAtlas does
2. Installation
3. Loading the package
4. Core API overview
5. English complex network text analysis with `keepwords_en`
6. Using `multiplex=True` to separate relationship layers
7. Italian complex network text analysis with `keepwords_ita`
8. English emotional analysis
9. Exporting and reusing networks
10. Troubleshooting
11. Citation and acknowledgements

---

## 1. What EmoAtlas does

EmoAtlas combines three ingredients:

1. **Natural language processing** through spaCy. spaCy tokenizes, lemmatizes, tags, and parses text.
2. **Textual forma mentis networks**. Relevant words are represented as nodes, and links represent relations extracted from the text and optional semantic enrichment.
3. **Emotion lexicons and Plutchik emotions**. Words are matched against multilingual emotion resources to compute emotional profiles over eight Plutchik emotions: anger, trust, surprise, disgust, joy, sadness, fear, and anticipation.

A typical workflow is:

```python
from emoatlas import EmoScores

emo = EmoScores(language="english", spacy_model="en_core_web_sm")
fmn = emo.formamentis_network(text)
emo.draw_formamentis(fmn)
emo.emotions(text)
emo.zscores(text)
```

---

## 2. Installation

### 2.1 Recommended Windows installation with the provided `.bat` file

This repository includes a Windows setup script named similar to:

```text
setup_emoatlas_windows.bat
```

The attached script automates the most important setup steps for Windows users with Anaconda or Miniconda. It:

1. Looks for `conda.bat` in common Anaconda and Miniconda locations.
2. Creates a Conda environment named `emoatlas311` with Python 3.11.
3. Activates the environment.
4. Upgrades `pip`, `setuptools`, and `wheel`.
5. Installs EmoAtlas from GitHub.
6. Repairs the `tokenizers` compatibility range.
7. Installs the lightweight English spaCy model `en_core_web_sm`.
8. Installs JupyterLab, Notebook, and IPython kernel support.
9. Downloads NLTK WordNet resources.
10. Registers a Jupyter kernel called `Python (emoatlas311)`.

Step-by-step:

1. Install Anaconda or Miniconda for Windows if you do not already have it.
2. Save the `.bat` file locally, for example on your Desktop or in the repository folder.
3. Double-click the `.bat` file, or run it from Anaconda Prompt.
4. Wait until the script prints `Setup complete`.
5. Open JupyterLab or Jupyter Notebook.
6. Choose the kernel named `Python (emoatlas311)`.

After the `.bat` installation, instantiate EmoAtlas with the same spaCy model installed by the script:

```python
from emoatlas import EmoScores

emo = EmoScores(language="english", spacy_model="en_core_web_sm")
```

Important: if you omit `spacy_model`, EmoAtlas tries to load the default large model for the selected language, for example `en_core_web_lg` for English and `it_core_news_lg` for Italian. The `.bat` file installs `en_core_web_sm`, so the examples below explicitly pass `spacy_model="en_core_web_sm"`.

### 2.2 Manual installation with pip

Use this route on macOS, Linux, Windows without Conda, or custom environments:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install git+https://github.com/MassimoStel/emoatlas.git
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
```

Then install NLTK resources:

```python
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

For better parsing quality, you may install large spaCy models instead:

```bash
python -m spacy download en_core_web_lg
python -m spacy download it_core_news_lg
```

Then initialize without passing `spacy_model`, or pass the large model explicitly:

```python
emo_en = EmoScores(language="english", spacy_model="en_core_web_lg")
emo_it = EmoScores(language="italian", spacy_model="it_core_news_lg")
```

### 2.3 Google Colab installation

In Colab, run shell commands with `!`:

```python
!pip install -q git+https://github.com/MassimoStel/emoatlas.git simplemma
!python -m spacy download en_core_web_sm
!python -m spacy download it_core_news_sm

import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

`simplemma` is optional, but useful for Italian because recent EmoAtlas code can apply conservative repairs to Italian lemmas when `simplemma` is available.

---

## 3. Loading the package

```python
from emoatlas import EmoScores
from emoatlas.formamentis_edgelist import keepwords_en, keepwords_it as keepwords_ita

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
```

The repository currently exposes the Italian keepword list as `keepwords_it`. The examples alias it as `keepwords_ita` because that name is clearer in bilingual tutorials:

```python
from emoatlas.formamentis_edgelist import keepwords_it as keepwords_ita
```

Initialize analyzers:

```python
emo_en = EmoScores(language="english", spacy_model="en_core_web_sm")
emo_it = EmoScores(language="italian", spacy_model="it_core_news_sm")
```

---

## 4. Core API overview

The main class is `EmoScores`.

| Function | Purpose |
| --- | --- |
| `EmoScores(language, spacy_model=None)` | Loads the emotion lexicon, spaCy model, antonym resources, baseline resources, and Plutchik emotion configuration. |
| `set_stemming_lemmatization("lemmatization" or "stemming")` | Switches between lemmatized and stemmed lexicon matching. Lemmatization is the default. |
| `set_baseline(baseline=None)` | Sets the baseline used for z-score emotional comparisons. |
| `emotions(obj, normalization_strategy="none", return_words=False)` | Counts Plutchik emotions in a text or forma mentis network. With `return_words=True`, returns the words matched to each emotion. |
| `zscores(obj, baseline=None, n_samples=300)` | Computes z-scores for the eight Plutchik emotions against a baseline distribution. |
| `formamentis_network(text, ...)` | Builds a textual forma mentis network from a text. |
| `draw_formamentis(fmn, ...)` | Visualizes a forma mentis network with valence-aware node and edge styling. |
| `extract_word_from_formamentis(fmn, target_word)` | Extracts the semantic frame around a target word from a full network. |
| `draw_formamentis_flower(text, ...)` | Builds a network and draws a Plutchik flower from the resulting network. |
| `draw_statistically_significant_emotions(obj, title=None)` | Draws a Plutchik flower showing statistically over- or under-represented emotions. |
| `draw_plutchik(scores, ...)` | Draws a Plutchik flower from user-provided scores. |
| `lemmatize_text(text)` | Returns the lemmatized words retained by the forma mentis pipeline. |
| `export_formamentis(fmnt, filename=None, path=None)` | Exports a non-multiplex edge list. |
| `import_formamentis(filepath)` | Imports a non-multiplex edge list. |
| `formamentis_to_nxgraph(fmnt)` | Converts a forma mentis network to a NetworkX graph. |
| `nxgraph_to_formamentis(graph)` | Converts a NetworkX graph to a forma mentis network object. |
| `combine_formamentis(edgelists, weights=False)` | Combines multiple networks or edge lists, optionally preserving edge weights. |
| `find_all_shortest_paths(graph, start_node, end_node)` | Finds all shortest conceptual paths between two words. |
| `get_top_quantile_shortest_paths(network, start_node, end_node, top_quantile=0.25)` | Keeps the strongest shortest paths from a weighted network. |
| `plot_mindset_stream(graph, start_node, end_node, ...)` | Visualizes shortest semantic paths between two concepts. |
| `calculate_path_weight(network, path)` | Computes the total weight of a path. |
| `export_whole_fmnt(fmnt, filename)` | Exports edges and nodes grouped by positive, negative, and neutral valence. |

---

## 5. English complex network text analysis with `keepwords_en`

The goal of this example is to build a rich English forma mentis network, inspect its layers, compute centrality, visualize the full network, extract a target semantic frame, and compute emotions on that frame.

### 5.1 Define a text

```python
text_en = """
A community facing uncertainty can still build hope, trust, and confidence.
Fear and fright may spread quickly when people feel alone, but support,
care, and clear communication can reduce anger and rage. The group learns
that resilience is not the absence of worry: resilience is the practice of
moving through concern together, protecting vulnerable people, and imagining
a safer future.
"""
```

### 5.2 Build the network

```python
from emoatlas import EmoScores
from emoatlas.formamentis_edgelist import keepwords_en

emo_en = EmoScores(language="english", spacy_model="en_core_web_sm")

fmnt_en = emo_en.formamentis_network(
    text_en,
    keepwords=keepwords_en,
    stopwords=[],
    max_distance=3,
    semantic_enrichment=["synonyms", "hypernyms"],
    multiplex=True,
)
```

What the main parameters mean:

- `keepwords=keepwords_en` keeps useful English discourse, pronoun, and semantic words that might otherwise be removed by default stopword filters.
- `stopwords=[]` means no additional custom stopwords are removed beyond spaCy stopword logic.
- `max_distance=3` links retained words that are within three syntactic steps inside the sentence-level dependency graph.
- `semantic_enrichment=["synonyms", "hypernyms"]` adds WordNet-based semantic links when available.
- `multiplex=True` keeps edge types separated instead of flattening them into a single edge list.

### 5.3 Inspect layer sizes and central words

```python
def flatten_fmnt_edges(fmnt):
    """Return a flat list of 2-tuples from a normal or multiplex forma mentis network."""
    if isinstance(fmnt.edges, dict):
        return [edge for layer_edges in fmnt.edges.values() for edge in layer_edges]
    return list(fmnt.edges)


def summarize_fmnt(fmnt, top_n=10):
    """Create a NetworkX graph and summary tables for a forma mentis network."""
    if isinstance(fmnt.edges, dict):
        layer_table = pd.DataFrame(
            [{"layer": layer, "n_edges": len(edges)} for layer, edges in fmnt.edges.items()]
        )
    else:
        layer_table = pd.DataFrame([{"layer": "all", "n_edges": len(fmnt.edges)}])

    graph = nx.Graph()
    graph.add_nodes_from(fmnt.vertices)
    graph.add_edges_from(flatten_fmnt_edges(fmnt))

    centrality = nx.closeness_centrality(graph) if graph.number_of_nodes() else {}
    centrality_table = (
        pd.DataFrame(
            [{"word": word, "closeness": score} for word, score in centrality.items()]
        )
        .sort_values("closeness", ascending=False)
        .head(top_n)
    )
    return graph, layer_table, centrality_table


G_en, layers_en, centrality_en = summarize_fmnt(fmnt_en)
print(layers_en)
print(centrality_en)
print("Nodes:", G_en.number_of_nodes(), "Edges:", G_en.number_of_edges())
```

### 5.4 Visualize the network

```python
emo_en.draw_formamentis(
    fmnt_en,
    layout="edge_bundling",
    thickness=1.5,
    alpha_syntactic=0.35,
    alpha_synonyms=0.85,
    alpha_hypernyms=0.45,
    hide_label=False,
)
plt.show()
```

### 5.5 Extract and analyze the semantic frame of a target word

```python
target = "resilience"

if target in fmnt_en.vertices:
    frame_en = emo_en.extract_word_from_formamentis(fmnt_en, target)
    emo_en.draw_formamentis(
        frame_en,
        highlight=[target],
        thickness=2,
        alpha_syntactic=0.45,
        alpha_synonyms=0.9,
        alpha_hypernyms=0.4,
    )
    plt.show()

    frame_text = " ".join(frame_en.vertices)
    print(emo_en.emotions(frame_text, return_words=True))
else:
    print(f"The target word {target!r} is not present after preprocessing. Try a displayed vertex instead.")
```

---

## 6. Using `multiplex=True` to separate relationship layers

A forma mentis network can contain different kinds of edges:

- **Syntactic edges**: extracted from the dependency structure of the text.
- **Synonym edges**: added through WordNet when two retained words are synonyms.
- **Hypernym edges**: added through WordNet when retained words are linked by hypernym/hyponym relations.

When `multiplex=False`, all these links are returned as one list. This is useful when you only need one graph, but it loses the edge type.

When `multiplex=True`, `fmnt.edges` becomes a dictionary. For example:

```python
{
    "syntactic": [...],
    "synonyms": [...],
    "hypernyms": [...]
}
```

### 6.1 Multiplex network with only synonym enrichment

Use this when you want to highlight syntactic links and synonym links separately, but do not want hypernym links.

```python
fmnt_synonyms_only = emo_en.formamentis_network(
    text_en,
    keepwords=keepwords_en,
    max_distance=3,
    semantic_enrichment="synonyms",
    multiplex=True,
)

for layer, edges in fmnt_synonyms_only.edges.items():
    print(f"{layer}: {len(edges)} edges")
```

Expected structure:

```text
syntactic: ... edges
synonyms: ... edges
```

Because `semantic_enrichment="synonyms"`, the output does not include a `hypernyms` layer. This is the cleanest setup for studying how lexical equivalence reinforces the syntactic structure of the text.

### 6.2 Visualize synonym links more strongly

```python
emo_en.draw_formamentis(
    fmnt_synonyms_only,
    layout="edge_bundling",
    thickness=2,
    alpha_syntactic=0.25,
    alpha_synonyms=0.95,
    alpha_hypernyms=0.0,
)
plt.show()
```

Interpretation:

- Use lower `alpha_syntactic` when you want the text-derived syntactic backbone to remain visible but subdued.
- Use higher `alpha_synonyms` when you want synonym relationships to stand out.
- Keep `alpha_hypernyms=0.0` in this example because no hypernym layer is requested.

---

## 7. Italian complex network text analysis with `keepwords_ita`

The Italian keepword list is exposed in the code as `keepwords_it`; this README aliases it as `keepwords_ita`.

### 7.1 Initialize Italian EmoAtlas

```python
from emoatlas import EmoScores
from emoatlas.formamentis_edgelist import keepwords_it as keepwords_ita

emo_it = EmoScores(language="italian", spacy_model="it_core_news_sm")
```

### 7.2 Define an Italian text

```python
text_it = """
Durante una crisi, una comunità può provare paura, timore e rabbia,
ma può anche costruire fiducia, cura e speranza. Le persone non sono sole:
parlano, ricordano, proteggono i più fragili e trasformano l'ansia in
attenzione condivisa. La paura non scompare, ma diventa parte di un percorso
verso coraggio, responsabilità e futuro.
"""
```

### 7.3 Build a multiplex Italian network

```python
fmnt_it = emo_it.formamentis_network(
    text_it,
    keepwords=keepwords_ita,
    stopwords=[],
    max_distance=3,
    semantic_enrichment=["synonyms", "hypernyms"],
    multiplex=True,
)

G_it, layers_it, centrality_it = summarize_fmnt(fmnt_it)
print(layers_it)
print(centrality_it)
print("Nodes:", G_it.number_of_nodes(), "Edges:", G_it.number_of_edges())
```

### 7.4 Visualize the Italian network

```python
emo_it.draw_formamentis(
    fmnt_it,
    layout="edge_bundling",
    thickness=1.5,
    alpha_syntactic=0.35,
    alpha_synonyms=0.85,
    alpha_hypernyms=0.45,
)
plt.show()
```

### 7.5 Extract the semantic frame around `paura`

```python
target_it = "paura"

if target_it in fmnt_it.vertices:
    frame_it = emo_it.extract_word_from_formamentis(fmnt_it, target_it)
    emo_it.draw_formamentis(
        frame_it,
        highlight=[target_it],
        thickness=2,
        alpha_syntactic=0.45,
        alpha_synonyms=0.9,
        alpha_hypernyms=0.4,
    )
    plt.show()

    frame_text_it = " ".join(frame_it.vertices)
    print(emo_it.emotions(frame_text_it, return_words=True))
else:
    print(f"The target word {target_it!r} is not present after preprocessing. Inspect fmnt_it.vertices.")
```

---

## 8. English emotional analysis

EmoAtlas can analyze emotions directly from raw text or from a forma mentis network. The direct route is useful for fast emotional profiling; the network route is useful when you want the emotional profile of a semantic frame or network-derived set of concepts.

### 8.1 Define an English emotional text

```python
emotion_text = """
I felt anxious and afraid at first, because the situation looked uncertain
and unfair. Yet the support of my friends gave me hope, trust, and joy.
The fear did not disappear completely, but it became easier to face with
care, patience, and confidence.
"""
```

### 8.2 Count emotion-bearing words

```python
emotion_words = emo_en.emotions(
    emotion_text,
    normalization_strategy="none",
    return_words=True,
)

emotion_words
```

The output is a dictionary indexed by Plutchik emotion. Each entry includes:

- `count`: how many distinct matched words are associated with that emotion after preprocessing;
- `words`: which words triggered that emotion.

### 8.3 Compute normalized emotion scores

```python
emotion_profile = emo_en.emotions(
    emotion_text,
    normalization_strategy="emotion_words",
    return_words=False,
)

pd.Series(emotion_profile).sort_values(ascending=False)
```

Normalization options:

- `"none"`: raw distinct emotion-word counts;
- `"text_length"`: counts divided by total preprocessed text length;
- `"emotion_words"`: counts divided by the total number of matched emotion words.

### 8.4 Compute z-scores

```python
z = emo_en.zscores(emotion_text, n_samples=300)
pd.Series(z).sort_values(ascending=False)
```

Z-scores compare the observed emotional distribution with a baseline. Positive values indicate emotions that are more represented than expected; negative values indicate emotions that are less represented than expected.

### 8.5 Draw a statistically significant Plutchik flower

```python
emo_en.draw_statistically_significant_emotions(
    emotion_text,
    title="English emotional profile"
)
plt.show()
```

The default significance threshold used by this wrapper is approximately `[-1.96, 1.96]`. Emotions above the upper threshold are interpreted as over-represented relative to the baseline; emotions below the lower threshold are interpreted as under-represented.

---

## 9. Exporting and reusing networks

### 9.1 Export a simple non-multiplex edge list

`export_formamentis()` does not support multiplex networks. Build a non-multiplex network first:

```python
fmnt_export = emo_en.formamentis_network(
    text_en,
    keepwords=keepwords_en,
    max_distance=3,
    semantic_enrichment="synonyms",
    multiplex=False,
)

emo_en.export_formamentis(fmnt_export, filename="english_formamentis_edges.txt")
```

### 9.2 Import a simple edge list

```python
fmnt_imported = emo_en.import_formamentis("english_formamentis_edges.txt")
```

### 9.3 Export a full network with valence categories

```python
emo_en.export_whole_fmnt(fmnt_export, "english_formamentis_full_export.txt")
```

The full export includes:

- edges;
- positive nodes;
- negative nodes;
- neutral nodes.

### 9.4 Shortest paths and mindset stream

```python
G_export = emo_en.formamentis_to_nxgraph(fmnt_export)

start_node = "fear"
end_node = "hope"

if start_node in G_export and end_node in G_export:
    paths = emo_en.find_all_shortest_paths(fmnt_export, start_node, end_node)
    print(paths)

    emo_en.plot_mindset_stream(
        graph=fmnt_export,
        start_node=start_node,
        end_node=end_node,
        title=f"Conceptual path: {start_node} to {end_node}",
        custom_font=14,
        figsize=(12, 8),
        marginset=0.25,
    )
    plt.show()
else:
    print("Choose start and end nodes from fmnt_export.vertices.")
```

---

## 10. Troubleshooting

### `ValueError: Can't find Spacy model ...`

Install the requested model, or pass the model you actually installed:

```bash
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
```

```python
EmoScores(language="english", spacy_model="en_core_web_sm")
EmoScores(language="italian", spacy_model="it_core_news_sm")
```

### WordNet or synonym enrichment returns no edges

Make sure NLTK resources are installed:

```python
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

Also remember that synonym and hypernym edges are only created when both related words are present in the final vertex list.

### My target word is missing from `fmnt.vertices`

EmoAtlas lemmatizes text. Your surface word may appear under a lemma. Inspect the retained vertices:

```python
sorted(fmnt_en.vertices)
```

Then use the displayed lemma as `target_word`.

### I used `multiplex=True` and NetworkX code failed

When `multiplex=True`, `fmnt.edges` is a dictionary, not a flat list. Flatten it before passing it to NetworkX:

```python
flat_edges = [edge for edges in fmnt.edges.values() for edge in edges]
G = nx.Graph()
G.add_nodes_from(fmnt.vertices)
G.add_edges_from(flat_edges)
```

### `export_formamentis()` fails on multiplex networks

Use `multiplex=False`, or export a specific layer manually:

```python
synonym_edges = fmnt_synonyms_only.edges.get("synonyms", [])
```

---

## 11. Citation and acknowledgements

If you use EmoAtlas in research, cite:

Semeraro, A., Vilella, S., Improta, R. et al. (2025). *EmoAtlas: An emotional network analyzer of texts that merges psychological lexicons, artificial intelligence, and network science*. Behavior Research Methods, 57, 77. https://doi.org/10.3758/s13428-024-02553-7

EmoAtlas is maintained and further developed by Prof. Massimo Stella. The original version was written by Dr. Alfonso Semeraro, with contributions to testing, debugging, and refactoring from Prof. Giulio Rossetti, Riccardo Improta and Dr. Finley Gibson.
