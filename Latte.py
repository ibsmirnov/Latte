import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import hdbscan
import seaborn as sns
import random
import time

class Latte:
    """
    LLM-Assisted Topic/Thematic Analysis (LATTE) class
    
    This class provides methods for analyzing common themes or topics 
    within text data while preserving researcher flexibility and control.
    """
    
    def __init__(self, data, attribute=None):
        """
        Initialize the Latte analyzer.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the text data to analyze.
            Must have a 'title' column, and optionally a 'text' column.
        
        attribute : str, optional
            The name of the column in data to use for grouping/comparison.
            For example, 'gender' to compare themes across genders.
        """
        # Store the original data
        self.data = data
        
        # Store the attribute name for grouping
        self.attribute = attribute
        
        # Verify that title column exists
        if 'title' not in data.columns:
            raise ValueError("DataFrame must contain a 'title' column")
        
        # Store titles
        self.titles = data['title'].fillna('').tolist()
        
        # Check if text column exists
        if 'text' in data.columns:
            # Create full_texts by combining title and text
            self.data['text'] = self.data['text'].fillna('')
            self.full_texts = self.data.apply(
                lambda row: row['title'] + ' ' + row['text'] if row['text'] else row['title'], 
                axis=1
            )
        else:
            # If no text column, full_texts are just titles
            self.full_texts = self.titles
        
        # Store attribute values if attribute was provided
        if attribute:
            if attribute not in data.columns:
                raise ValueError(f"DataFrame does not contain the specified attribute column: {attribute}")
            self.attribute_values = data[attribute].tolist()
        else:
            self.attribute_values = None
            
        # Initialize embeddings as None
        self.embeddings = None

        self.red = '#ae2012'
        self.blue = '#669bbc'
        
    def default_embedding_function(self, texts):
        """
        Default embedding function using a local model from Hugging Face.
        
        Parameters:
        -----------
        texts : list of str
            List of texts to embed.
            
        Returns:
        --------
        numpy.ndarray
            Array of embeddings, one per text.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "The sentence-transformers package is required for default embeddings. "
                "Install it with: pip install sentence-transformers"
            )
        
        # Use all-MiniLM-L6-v2 as a good default model (fast and reasonably accurate)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return model.encode(texts)
    
    def embed(self, embedding_function=None, texts=None):
        """
        Generate embeddings for the texts using the provided embedding function.
        If no embedding function is provided, use the default local model.
        
        Parameters:
        -----------
        embedding_function : callable, optional
            A function that takes a list of texts and returns embeddings.
            If None, the default embedding function will be used.
            
        texts : list of str, optional
            Texts to embed. If None, full_texts will be used.
            
        Returns:
        --------
        numpy.ndarray
            Array of embeddings, one per text.
        
        Note:
        -----
        The embeddings are also stored in self.embeddings for later use.
        """
        if texts is None:
            # Check if full_texts is a list or pandas Series/DataFrame
            if hasattr(self.full_texts, 'tolist'):
                texts = self.full_texts.tolist()
            else:
                texts = self.full_texts
        
        if embedding_function is None:
            print("Using default embedding model (all-MiniLM-L6-v2)...")
            embedding_function = self.default_embedding_function
        
        # Generate embeddings
        self.embeddings = embedding_function(texts)
        
        print(f"Generated embeddings with shape: {self.embeddings.shape}")

        return self


    def reduce(self, n_neighbors: int = 15):
        """
        Reduce the dimensionality of embeddings using UMAP.
        
        Parameters:
        -----------
        n_neighbors : int, optional (default=15)
            Number of neighbors to consider for each point in UMAP.
            Lower values emphasize local structure, higher values global structure.
            
        Note:
        -----
        The reduced embeddings are stored in self.reduced_embeddings for
        visualization and further analysis.
        """
        # Convert embeddings to numpy array if not already
        embeddings_array = np.array(self.embeddings)
        
        # Create and configure UMAP reducer
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,  # Number of neighbors for local structure
            n_components=2,           # Reduce to 2D for visualization
            random_state=1,           # For reproducibility
            n_jobs=1                  # Number of parallel jobs
        )
   
        # Perform dimensionality reduction
        self.reduced_embeddings = reducer.fit_transform(embeddings_array)
        
        print(f"Reduced embeddings to shape: {self.reduced_embeddings.shape}")

        return self

    def plot_embeddings(self, color: bool = True, marker_size: float = 20, alpha: float = 0.5):
       """
       Plot reduced dimensions as a minimalistic scatter plot.
       Points colored based on binary attribute if provided.
       
       Args:
           marker_size: Size of markers in scatter plot
           alpha: Transparency of markers
       """
       # Set up the figure
       plt.figure(figsize=(6, 6))

       # Define colors
       if color:
           color_0 = self.blue
           color_1 = self.red
       else:
           color_0 = '#000000'
           color_1 = '#000000'

       # If binary attribute exists, use it for colors
       if self.attribute_values is not None:
           colors = [color_1 if x else color_0 for x in self.attribute_values]
       else:
           colors = color_0
           
       # Create scatter plot
       plt.scatter(
           self.reduced_embeddings[:, 0],
           self.reduced_embeddings[:, 1],
           s=marker_size,
           alpha=alpha,
           c=colors,
           edgecolors='none'  # This removes the marker border
       )
       
       # Remove all decorations
       plt.axis('off')
       
       # Ensure tight layout
       plt.tight_layout()

    def cluster(self, min_cluster_size: int = 5, min_samples: int = None, cluster_selection_epsilon: float = 0.0):
        """
        Perform clustering using HDBSCAN and save its condensed tree and labels.
        Also instantiate a helper object for hierarchical exploration.
        """
        data_array = np.array(self.reduced_embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=cluster_selection_epsilon
        )
        clusterer.fit(data_array)

        self.clusterer = clusterer
        
        self._compute_tree()

        self._compute_fractions()

        return self
    
    def _compute_tree(self):
        df = self.clusterer.condensed_tree_.to_pandas()
        
        self.clusters = {}
        
        for i, row in df.iterrows():
            parent = int(row['parent'])
            if parent not in self.clusters:
                self.clusters[parent] = {"id": parent, "parent": -1,  "children": [], "points": [], "level": 0}
        
            child = int(row['child'])
        
            if row['child_size'] == 1:
                self.clusters[parent]['points'].append(child)
            else:
                if child not in self.clusters:
                    self.clusters[child] = {"id": child, "parent": -1,  "children": [], "points": [], "level": row['lambda_val']}
            
                self.clusters[parent]['children'].append(child)
                self.clusters[child]['parent'] = parent

        ls = [self.clusters[x]['level'] for x in self.clusters]
        ls = sorted(list(set(ls)))
        ls_map = {l: len(ls) - i - 1 for i, l in enumerate(ls)}

        for x in self.clusters:
            self.clusters[x]['level'] = ls_map[self.clusters[x]['level']]
            
        root = [cl for cl in self.clusters if self.clusters[cl]['parent'] == -1]
        
        if len(root) > 1:
            raise ValueError("Assumption of a single root in a tree is violated")
        root = root[0]
        
        self._get_points(root)

    def _get_points(self, idn):
        cl = self.clusters[idn]
    
        if cl['children']:
            total = []
            for child in cl['children']:
                total += self._get_points(child)
            cl['points'] += total
          
        return cl['points']
    
    def _compute_fractions(self):
        """
        Compute both raw and normalized fractions of points with attribute=1 for each cluster.
        
        For each cluster, computes:
        - fraction: raw fraction of 1 values in the cluster
        - fraction_norm: normalized difference from the overall fraction, scaled to [-1, 1]
          
        The normalization formula is:
        - If p < P: (p - P) / P
        - If p >= P: (p - P) / (1 - P)
        
        where:
        p = cluster fraction
        P = overall fraction across all points
        
        This ensures that deviations are properly scaled relative to how much room there is to deviate
        in either direction from the overall fraction.
        """
        if self.attribute_values is None:
            return
        
        # Calculate overall fraction P
        total_true = sum(1 for x in self.attribute_values if x)
        total_points = len(self.attribute_values)
        overall_fraction = total_true / total_points
            
        for cluster_id in self.clusters:
            cluster = self.clusters[cluster_id]
            points = cluster['points']
            if points:  # Only compute if cluster has points
                true_count = sum(1 for point in points if self.attribute_values[point])
                cluster_fraction = true_count / len(points)
                
                # Store raw proportion
                cluster['fraction'] = cluster_fraction
                
                # Calculate normalized proportion
                if cluster_fraction < overall_fraction:
                    normalized = (cluster_fraction - overall_fraction) / overall_fraction
                else:
                    normalized = (cluster_fraction - overall_fraction) / (1 - overall_fraction)
                    
                cluster['fraction_norm'] = normalized

    def _get_level_clusters(self, level: int = 0):
        """
        Get clusters at a specific level of aggregation.
        
        Args:
            level: Level of aggregation to retrieve
        """

        level_clusters = []

        for cl_id in self.clusters:
            cl = self.clusters[cl_id]

            # We don't include clusters that are below the level threshold
            if cl['level'] < level: continue

            # We also need to skip clusters with children above or at the lelvel threshold

            if cl['children']:
                child = cl['children'][0]
                if self.clusters[child]['level'] >= level: continue

            level_clusters.append(cl)  

        return level_clusters  

    def print_clusters(self, level: int = 0):
        clusters_to_print = self._get_level_clusters(level)

        for cluster in clusters_to_print:
            print('=' * 3, 'Cluster', cluster['id'], f'({len(cluster["points"])} items)', '=' * 3)
            if 'fraction_norm' in cluster:
                # Format fraction_norm as percentage with sign
                fraction_val = cluster['fraction_norm']
                sign = '+' if fraction_val > 0 else ''
                print(f"Leaning: {sign}{fraction_val:.2f}")
            if 'annotation' in cluster:
                print(cluster['annotation'])
            print()
            
            # Display up to 5 titles
            titles_to_show = min(5, len(cluster['points']))
            for i in range(titles_to_show):
                print(self.titles[cluster['points'][i]])
            
            # Add ellipsis if there are more titles
            if len(cluster['points']) > 5:
                print('...')
            
            print()
            print()

    def plot_clusters(self, level: int = 0, marker_size: float = 20, alpha: float = 0.5, show_contours: bool = False, fill_contours: bool = False, ax=None):
        """
        Plot clusters by coloring points belonging to leaf clusters (clusters with no children), or
        to higher level clusters if level parameter is provided and is larger than 0. Level corresponds
        to different values of lambda at which aggregation of two clusters is happening.
        
        Args:
            level: Level of aggregation to display
            marker_size: Size of markers in scatter plot
            alpha: Transparency of markers
            show_contours: Whether to show contour lines around clusters
            fill_contours: Whether to fill contours with color based on attribute fraction
            ax: Matplotlib axis to plot on. If None, creates a new figure.
        
        Returns:
            The matplotlib axis with the plot
        """
        # Create new figure if no axis is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        noise_mask = self.clusterer.labels_ == -1

        # Plot noise points in light gray
        ax.scatter(
            self.reduced_embeddings[noise_mask, 0],
            self.reduced_embeddings[noise_mask, 1],
            c='lightgray',
            alpha=alpha,
            s=marker_size,
            edgecolors='none'
        )   

        # Plot all clustered points in black
        clustered = self.reduced_embeddings[~noise_mask]
        ax.scatter(
            clustered[:, 0],
            clustered[:, 1],
            c='black',
            alpha=alpha,
            s=marker_size,
            edgecolors='none'
        )

        # Draw contours if enabled
        if show_contours:
            clusters_to_display = self._get_level_clusters(level)

            for i, cluster in enumerate(clusters_to_display):
                if len(cluster['points']) > 0:  # Only draw contours for clusters with enough points
                    # Get points for this cluster
                    cluster_points = self.reduced_embeddings[cluster['points']]
                    
                    # Draw contour using KDE plot
                    sns.kdeplot(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        levels=[0.1],  # Single contour level
                        color='black',
                        linestyles='--',
                        alpha=1,
                        zorder=-1,
                        linewidths=0.5,
                        ax=ax
                    )

                    if fill_contours:
                        f = cluster['fraction_norm']
                        if f > 0:
                            color = self.red
                        else:
                            color = self.blue

                        sns.kdeplot(
                            x=cluster_points[:, 0],
                            y=cluster_points[:, 1],
                            levels=[0.1, 1.0],
                            fill=True,
                            color=color,
                            alpha=abs(f),
                            zorder=-2,
                            ax=ax
                        )
        
        # Remove all decorations
        ax.axis('off')
        
        # Return the axis for further customization
        return ax

    def annotate(self, level=0, max_titles=50, llm_function=None, prompt=None):
        """
        Annotate clusters using an LLM (either a custom function or the default TinyLlama model).
        
        Args:
            level: Level of aggregation to annotate
            max_titles: Maximum number of titles to include in each prompt
            llm_function: Custom function for LLM inference, should accept a prompt and return a response
            prompt: Custom prompt template to use for annotation
            verbose: Whether to print progress information
            
        Returns:
            Dictionary mapping cluster IDs to annotations
        """
        
        # Set default prompt if none provided
        if prompt is None:
            prompt = """I have grouped text published on an online social media platform, using cluster analysis.
Each cluster represents a set of texts that are thematically similar.
Your task is to identify the main topic or a couple of key topics for the following cluster of texts.
Please, return a concise annotation for the following texts.
Avoid any introductions (such as "The main topics of this cluster of questions are:") and directly name the main topic or main topics.
Avoid any markdown.
Ensure that your answer is no longer than one sentence.
Please identify the main topic or topics for the following texts from the cluster::\n{texts}"""
        
        # Initialize annotations dictionary
        self.annotations = {}
        
        # If no custom LLM function is provided, use the default TinyLlama
        if llm_function is None:
            llm_function = self._default_llm_function()
        
        # Process each cluster
        for i, cluster_id in enumerate(self.clusters):
            cluster = self.clusters[cluster_id]
            points = cluster['points']
            
            if len(points) == 0:
                continue
                
            # Get titles for this cluster
            cluster_titles = [self.titles[point_idx] for point_idx in points]
            
            # Randomly sample if there are too many titles
            if len(cluster_titles) > max_titles:
                sampled_titles = random.sample(cluster_titles, max_titles)
            else:
                sampled_titles = cluster_titles
            
            # Join titles into a single text
            texts = "\n\n".join(sampled_titles)
            
            # Format the prompt
            formatted_prompt = prompt.format(texts=texts)
            
            print(f"\nAnnotating cluster {i+1}/{len(self.clusters)} (ID: {cluster_id}) with {len(sampled_titles)} titles")
            
            # Get annotation from LLM
            annotation = llm_function(formatted_prompt)
            
            # Store the annotation
            self.annotations[cluster_id] = annotation
            
            # Add annotation to the cluster data structure
            self.clusters[cluster_id]['annotation'] = annotation
    
    def _default_llm_function(self):
        """
        Create and return the default LLM function using Qwen2.5-7B-Instruct.
        
        Returns:
            Function that accepts a prompt and returns a response
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "The transformers package is required for default LLM. "
                "Install it with: pip install transformers"
            )
        
        # Load model and tokenizer
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        print(f"Loading LLM model: {model_name}")
        
        # Cache for model and tokenizer to avoid reloading for each cluster
        if not hasattr(self, '_llm_model') or not hasattr(self, '_llm_tokenizer'):
            # Load model
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Load tokenizer
            self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use cached model and tokenizer
        model = self._llm_model
        tokenizer = self._llm_tokenizer
        
        def inference_function(prompt):
            """Inner function to run inference with Qwen2.5-7B."""
            # Format messages using the official approach
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that identifies the main topic or key topics from a set of texts. You must provide only the main topics in a single, concise sentence. Do not add any introductions or explanations."
                },
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Prepare model inputs
            model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
            
            # Generate response
            start_time = time.time()
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
                temperature=0.3,
                top_p=0.9
            )
            
            # Extract only the newly generated tokens
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode the response
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            inference_time = time.time() - start_time
            
            print(f"Inference took {inference_time:.2f} seconds")
            
            return response
        
        return inference_function