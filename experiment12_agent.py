import os
import re
import json
import heapq
from typing import List, Dict, Tuple, Optional, Any

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

class SentimentModule:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    def analyze(self, text: str) -> str:
        score = self.analyzer.polarity_scores(text)["compound"]
        if score > 0.3: return "That sounds Positive! 😊"
        elif score < -0.3: return "That sounds Negative. 😞"
        else: return "That sounds Neutral. 😐"

class KnowledgeRetriever:
    def __init__(self, documents: List[str]):
        self.docs = [doc.strip() for doc in documents if doc.strip()]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.vectors = self.vectorizer.fit_transform(self.docs) if self.docs else None
    def retrieve(self, query: str, top_k: int = 1) -> List[str]:
        if not self.docs: return ["The knowledge base is empty."]
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.vectors)[0]
        top_indices = np.argsort(-sims)[: min(top_k, len(self.docs))]
        return [self.docs[i] for i in top_indices]

class FamilyReasoner:
    def __init__(self, relations: Dict[str, List[str]]):
        self.relations = {k.lower(): [name.capitalize() for name in v] for k, v in relations.items()}
    def get_children(self, parent: str) -> List[str]:
        return self.relations.get(parent.lower(), [])
    def get_grandchildren(self, grandparent: str) -> List[str]:
        children = self.get_children(grandparent)
        if not children: return []
        grandchildren = []
        for child in children:
            grandchildren.extend(self.get_children(child))
        return grandchildren

class BayesianTrafficAdvisor:
    def __init__(self):
        self.prior_delay = 0.2
        self.traffic_effects = {"low": 0.1, "medium": 0.4, "high": 0.8}
        self.weather_effects = {"clear": 0.1, "rainy": 0.5, "stormy": 0.7}
    def estimate_delay(self, traffic_level: str = "medium", weather: str = "clear") -> float:
        p_traffic = self.traffic_effects.get(traffic_level, 0.4)
        p_weather = self.weather_effects.get(weather, 0.1)
        posterior = self.prior_delay * p_traffic * p_weather * 50 # Heuristic scaling
        return min(posterior, 1.0)

class AStarPlanner:
    def __init__(self, graph: Dict[str, Dict[str, float]], heuristics: Dict[str, Dict[str, float]]):
        self.graph = graph
        self.heuristics = heuristics
    def plan(self, start: str, goal: str) -> Tuple[Optional[float], Optional[List[str]]]:
        open_set = [(self.heuristics[start][goal], 0, start, [start])]
        visited = set()
        while open_set:
            _, g_cost, current_node, path = heapq.heappop(open_set)
            if current_node == goal: return g_cost, path
            if current_node in visited: continue
            visited.add(current_node)
            for neighbor, weight in self.graph.get(current_node, {}).items():
                if neighbor not in visited:
                    new_g_cost = g_cost + weight
                    f_cost = new_g_cost + self.heuristics[neighbor][goal]
                    heapq.heappush(open_set, (f_cost, new_g_cost, neighbor, path + [neighbor]))
        return float("inf"), []

class SmartAgent:
    def __init__(self):
        kb_docs = self._load_text_file("kb.txt")
        family_data = self._load_json_file("family.json")
        map_data = self._load_json_file("map.json")
        
        print("Initializing modules...")
        self.sentiment = SentimentModule()
        self.retriever = KnowledgeRetriever(kb_docs)
        self.family = FamilyReasoner(family_data if family_data else {})
        self.traffic = BayesianTrafficAdvisor()
        
        graph = map_data.get("graph", {}) if map_data else {}
        heuristics = map_data.get("heuristics", {}) if map_data else {}
        self.planner = AStarPlanner(graph, heuristics)
        print("Agent is ready. Ask me for 'help' to see what I can do!")

    def _load_text_file(self, filepath: str) -> List[str]:
        try:
            with open(filepath, "r") as f: return f.readlines()
        except FileNotFoundError: return []

    def _load_json_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        try:
            with open(filepath, "r") as f: return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError): return None

    def handle_input(self, query: str) -> str:
        q = query.lower()

        patterns = {
            r"list family|who is in the family tree": (self._handle_list_family, 'none'),
            r"list locations|show locations|what places do you know": (self._handle_list_locations, 'none'),
            r"(?:shortest )?(?:path|route|way) from (\w+) to (\w+)": (self._handle_path, 'groups'),
            r"(?:grandchildren|grandkids) of (\w+)": (self._handle_grandchildren, 'groups'),
            r"(?:children|kids) of (\w+)": (self._handle_children, 'groups'),
            r"traffic|delay": (self._handle_traffic, 'query'),
            r"tell me about|what is|explain": (self._handle_retrieval, 'query'),
            r"feeling|i feel|mood": (self.sentiment.analyze, 'query'),
            r"help": (self._handle_help, 'none'),
        }

        for pattern, (handler, arg_type) in patterns.items():
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                if arg_type == 'groups': return handler(*match.groups())
                elif arg_type == 'query': return handler(query)
                else: return handler()
        
        return self._handle_fallback(query)

    def _handle_fallback(self, query: str) -> str:
        sentiment_result = self.sentiment.analyze(query)
        if "Neutral" not in sentiment_result:
            return sentiment_result
        else:
            return "I'm not sure about that. Try asking me for 'help' to see my capabilities."
    
    def _handle_retrieval(self, query: str) -> str:
        clean_query = re.sub(r"tell me about|what is|explain", "", query, flags=re.IGNORECASE).strip()
        if len(clean_query) < 3:
            return "Please be more specific. What would you like to know about?"
        return "\n".join(self.retriever.retrieve(query))

    def _handle_children(self, parent: str) -> str:
        if parent.lower() not in self.family.relations:
            return f"Sorry, the name '{parent.capitalize()}' is not in my family tree."
        children = self.family.get_children(parent)
        return f"The children of {parent.capitalize()} are: {', '.join(children)}." if children else f"{parent.capitalize()} has no children in my records."

    def _handle_grandchildren(self, grandparent: str) -> str:
        if grandparent.lower() not in self.family.relations:
            return f"Sorry, the name '{grandparent.capitalize()}' is not in my family tree."
        grandchildren = self.family.get_grandchildren(grandparent)
        return f"The grandchildren of {grandparent.capitalize()} are: {', '.join(grandchildren)}." if grandchildren else f"{grandparent.capitalize()} has no grandchildren in my records."
    
    def _handle_traffic(self, query: str) -> str:
        traffic_m = re.search(r"(low|medium|high) traffic", query)
        weather_m = re.search(r"(clear|rainy|stormy) weather", query)
        traffic = traffic_m.group(1) if traffic_m else "medium"
        weather = weather_m.group(1) if weather_m else "clear"
        prob = self.traffic.estimate_delay(traffic, weather)
        return f"With {traffic} traffic and {weather} weather, I estimate a {prob*100:.0f}% chance of significant delays."

    def _handle_path(self, start: str, goal: str) -> str:
        start_node, goal_node = start.upper(), goal.upper()
        if start_node not in self.planner.graph:
            return f"I don't know the location '{start_node}'. Try 'list locations' to see where I can navigate."
        if goal_node not in self.planner.graph:
            return f"I don't know the location '{goal_node}'. Try 'list locations' to see where I can navigate."
        
        cost, path = self.planner.plan(start_node, goal_node)
        return f"The best path from {start_node} to {goal_node} is: {' -> '.join(path)} (Total Cost: {cost})" if path else f"Sorry, I couldn't find a path from {start_node} to {goal_node}."

    def _handle_list_family(self) -> str:
        names = sorted([name.capitalize() for name in self.family.relations.keys()])
        return f"My family tree includes: {', '.join(names)}."

    def _handle_list_locations(self) -> str:
        locations = sorted(self.planner.graph.keys())
        return f"I know these locations: {', '.join(locations)}."
        
    def _handle_help(self) -> str:
        return """
You can ask me things like:
- **Knowledge**: "Tell me about machine learning"
- **Family**: "children of alice" or "list family"
- **Pathfinding**: "path from a to d" or "list locations"
- **Traffic**: "What's the delay with high traffic and rainy weather?"
- **Sentiment**: "I had a wonderful day"
"""

def main_cli():
    print("=" * 40)
    print("=== Smart Personal Assistant Agent ===")
    print("=" * 40)
    agent = SmartAgent()
    while True:
        try:
            txt = input("You> ")
            if not txt or not txt.strip():
                continue
            if txt.lower() in ["exit", "quit"]:
                print("Agent> Goodbye!")
                break
            resp = agent.handle_input(txt)
            print(f"Agent> {resp}")
        except (KeyboardInterrupt, EOFError):
            print("\nAgent> Goodbye!")
            break

if __name__ == "__main__":
    main_cli()
