import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

class SemanticEntropyComparer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def compare(self, text1, text2):
        # 1. Generate embeddings
        embeddings = self.model.encode([text1, text2])
        
        # 2. Calculate Cosine Similarity
        similarity = 1 - cosine(embeddings[0], embeddings[1])
        
        # 3. Calculate "Semantic Entropy"
        entropy = self._calculate_shannon_entropy(similarity)
        
        return {
            "similarity_score": round(float(similarity), 4),
            "semantic_entropy": round(float(entropy), 4),
            "verdict": "Redundant" if similarity > 0.8 else "Distinct"
        }

    def _calculate_shannon_entropy(self, similarity):
        p = np.clip((1 + similarity) / 2, 1e-9, 1 - 1e-9)  # Normalise to [0,1]
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

if __name__ == "__main__":
    comparer = SemanticEntropyComparer()

    str1 = """Direct Answer
    The primary cause of contemporary climate change is the increase in atmospheric greenhouse gases (GHGs) resulting largely from human activities. These gases trap heat in the Earth’s atmosphere through the Greenhouse Effect, leading to long-term warming of the climate system.

    1. Fossil Fuel Combustion

    The largest contributor is the burning of fossil fuels such as coal, oil, and natural gas for electricity generation, transportation, and industrial processes. This releases substantial amounts of carbon dioxide (CO₂) into the atmosphere. According to the Intergovernmental Panel on Climate Change (IPCC), fossil fuel use accounts for the majority of anthropogenic CO₂ emissions globally.

    2. Deforestation and Land-Use Change

    Forests act as carbon sinks by absorbing CO₂ through Photosynthesis. When forests are cleared for agriculture, urban development, or logging, stored carbon is released and the capacity for carbon absorption declines. This amplifies atmospheric greenhouse gas concentrations.

    3. Agriculture and Livestock

    Agricultural activities produce significant quantities of methane (CH₄) and nitrous oxide (N₂O). Methane is emitted during livestock digestion and rice cultivation, while nitrous oxide arises from fertilizer use. Both gases have higher warming potential than CO₂ per unit mass.

    4. Industrial Processes

    Manufacturing sectors, including cement production and chemical industries, emit greenhouse gases directly. Some processes release fluorinated gases, which are potent and long-lived contributors to atmospheric warming.

    5. Waste Management

    Landfills and wastewater treatment generate methane through the decomposition of organic waste under anaerobic conditions.

    Conclusion

    Scientific consensus indicates that modern climate change is primarily driven by human-induced greenhouse gas emissions—especially from fossil fuel combustion, deforestation, agriculture, and industry. Assessments by organizations such as the Intergovernmental Panel on Climate Change conclude that these anthropogenic factors are responsible for the majority of observed global warming since the mid-20th century."""
    str2 = """Direct Answer

    The primary cause of modern climate change is the increase in greenhouse gases in the atmosphere due to human activities. These gases trap heat, intensifying the natural Greenhouse Effect and raising global temperatures.

    Main Causes
    1. Fossil Fuel Combustion

    Burning coal, oil, and natural gas for electricity, transportation, and industry releases large amounts of carbon dioxide (CO₂). This is the largest contributor to anthropogenic climate change. Industrialization since the 19th century has significantly increased atmospheric CO₂ levels.

    2. Deforestation and Land-Use Change

    Forests absorb CO₂ through Photosynthesis. When forests are cleared for agriculture, urbanization, or logging, two effects occur:

    Stored carbon is released into the atmosphere.

    The planet’s capacity to absorb CO₂ declines.
    This process contributes significantly to global emissions.

    3. Agriculture and Livestock

    Agriculture produces greenhouse gases such as:

    Methane (CH₄) from cattle digestion and manure.

    Nitrous oxide (N₂O) from fertilizers and soil management.
    These gases have stronger heat-trapping potential than CO₂ over shorter time periods.

    4. Industrial Processes

    Manufacturing, cement production, and chemical industries release CO₂ and other gases such as fluorinated compounds. These gases often have very high global warming potential.

    5. Waste Management

    Landfills produce methane as organic waste decomposes without oxygen, adding to atmospheric greenhouse gases.

    Context and Limitations

    Natural factors—such as volcanic activity or solar variability—can influence climate. However, current scientific assessments, including those by the Intergovernmental Panel on Climate Change, conclude that human activities are the dominant driver of warming since the mid-20th century.

    Sources

    Intergovernmental Panel on Climate Change (IPCC), AR6 Climate Change 2021–2023

    Food and Agriculture Organization (FAO), State of the World’s Forests

    U.S. Environmental Protection Agency (EPA), Sources of Greenhouse Gas Emissions"""

    results = comparer.compare(str1, str2)
    print(f"Results: {results}")