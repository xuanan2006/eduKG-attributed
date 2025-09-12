import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import openai 
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import tqdm
from typing import List, Dict, Any, Optional, Tuple
import copy
import glob
import re
import datetime
# import torch
from rank_bm25 import BM25Okapi
# --- 0. Configuration and Setup ---

# CHOOSE YOUR LLM PROVIDER: "openai" or "gemini"
LLM_PROVIDER = "openai"

# --- API KEYS (REPLACE WITH YOURS) ---
# It's recommended to use environment variables for security
OPENAI_API_KEY = "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"
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"  # Replace if using Gemini

# --- MODEL CONFIGURATION ---
OPENAI_EXTRACTION_MODEL = "gpt-4o-mini"
OPENAI_DEFINITION_MODEL = "gpt-4o-mini"
OPENAI_VERIFICATION_MODEL = "gpt-4o-mini"
OPENAI_REJECTION_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-1.5-flash"  # Changed to 1.5-flash as 2.0 is not a valid model name

# --- FILE PATHS ---
INPUT_FOLDER = r"E:\Bean\hcmut\documents (real)\URA\CCE eduKG\chuẩn hóa dư liệu\subsets of chunked data\5pct_size_head_path"
LOGS_FOLDER_BASE = r"E:\Bean\hcmut\documents (real)\URA\CCE eduKG\KG construction\logs"
PROMPTS_FOLDER_BASE = r"E:\Bean\hcmut\documents (real)\URA\CCE eduKG\KG construction\prompts_and_answers"
INITIAL_KG_FILE = "initial_kg.json"
FINAL_KG_FILE = r"E:\Bean\hcmut\documents (real)\URA\CCE eduKG\KG construction\5pt_size_v1"

# --- ALGORITHM TUNING --- 
# Cần sửa: Từ threshold thành top k
SCHEMA_SIMILARITY_THRESHOLD = 0.60
INSTANCE_SIMILARITY_THRESHOLD = 0.85
#Trình tự thực thi mới: 
#    BM25 -> top 1000 
# -> Bi-encoder -> top 100 
# -> Cross-encoder -> top 5/10 
# -> Decision
TOP_K_BM25 = 1000
TOP_K_BI_ENCODER = 100
TOP_K_CROSS_ENCODER = 10

# --- Global LLM Clients and Logging ---
openai_client = None
CURRENT_FILE_LLM_LOGS: List[Dict[str, Any]] = []

# --- Setup and Validation ---
if LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY or "YOUR_API_KEY_HERE" in OPENAI_API_KEY:
        print("Error: Please set your OPENAI_API_KEY to use the 'openai' provider.")
        exit()
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    print("Using OpenAI provider.")
elif LLM_PROVIDER == "gemini":
    if not GOOGLE_API_KEY or "YOUR_GOOGLE_API_KEY_HERE" in GOOGLE_API_KEY:
        print("Error: Please set your GOOGLE_API_KEY to use the 'gemini' provider.")
        exit()
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Using Gemini provider.")
else:
    print(f"Error: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Please choose 'openai' or 'gemini'.")
    exit()


# --- Utility Functions ---

def call_llm(prompt: str, provider: str, model_name: str, max_tokens: Optional[int] = None) -> str:
    """
    A unified function to call either OpenAI or Gemini models.
    It also logs every prompt, response, and token usage to a global list.
    """
    response_content = ""
    logged_response = ""
    usage_info = {"input_tokens": 0, "output_tokens": 0}

    try:
        if provider == "openai":
            response = openai_client.chat.completions.create(
                #Gợi ý: Có thể dùng client.responses.create()
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens
            )
            response_content = response.choices[0].message.content.strip()
            if response.usage:
                usage_info["input_tokens"] = response.usage.prompt_tokens
                usage_info["output_tokens"] = response.usage.completion_tokens
            logged_response = response_content

        elif provider == "gemini":
            model = genai.GenerativeModel(model_name)
            generation_config = GenerationConfig(
                temperature=0.0,
                max_output_tokens=max_tokens if max_tokens is not None else 8192
            )
            response = model.generate_content(prompt, generation_config=generation_config)

            if response.parts:
                response_content = response.text.strip()
            else:
                response_content = ""

            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_info["input_tokens"] = response.usage_metadata.prompt_token_count
                usage_info["output_tokens"] = response.usage_metadata.candidates_token_count

            logged_response = response_content

    except Exception as e:
        error_message = f"Error calling LLM API: {e}"
        print(error_message)
        logged_response = f"LLM_CALL_ERROR: {str(e)}"

    CURRENT_FILE_LLM_LOGS.append({
        "prompt": prompt,
        "response": logged_response,
        "usage": usage_info
    })

    return response_content


def clean_llm_json_output(text: str) -> str:
    """Removes markdown code fences from LLM JSON output."""
    # This regex is more robust for handling various markdown formats
    match = re.search(r"```(json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(2).strip()
    return text.strip()


def _create_id_from_name(name: str) -> str:
    """Creates a standardized ID from a name string."""
    s = name.lower()
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[đ]', 'd', s)
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')


# --- NEW: Non-destructive property merging function ---
def merge_properties_uniquely(target_props: Dict[str, Any], source_props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges properties from a source dictionary into a target dictionary non-destructively.
    If a key from the source already exists in the target with a different value,
    it adds the new value with a suffixed key (e.g., 'location_2').
    """
    if not isinstance(target_props, dict): target_props = {}
    if not source_props or not isinstance(source_props, dict): return target_props

    for key, value in source_props.items():
        if key not in target_props:
            target_props[key] = value
        elif target_props[key] == value:
            continue
        else:
            i = 2
            new_key = f"{key}_{i}"
            while new_key in target_props:
                i += 1
                new_key = f"{key}_{i}"
            print(f"        -> Conflict for property '{key}'. Preserving original. Adding new as '{new_key}'.")
            target_props[new_key] = value
    return target_props


try:
    with open(INITIAL_KG_FILE, "r", encoding="utf-8") as f:
        INITIAL_KNOWLEDGE_GRAPH = json.load(f)
except FileNotFoundError:
    print(f"Warning: Initial knowledge graph file '{INITIAL_KG_FILE}' not found. Starting with an empty graph.")
    INITIAL_KNOWLEDGE_GRAPH = {
        "classes": [], "class_definitions": [], "instances": [],
        "relationship_types": [], "relationship_definitions": [], "relationships": []
    }
except json.JSONDecodeError:
    print(f"Error: Could not parse JSON from '{INITIAL_KG_FILE}'. Please check the file format.")
    exit()


def extract_graph_from_text(text: str, existing_classes: List[str]) -> Dict:
    """Uses an LLM to perform OIE and extract instances, their properties, and relationships with properties."""
    print("--- Stage 1: Extracting Open Graph from Text ---")

    model_map = {"openai": OPENAI_EXTRACTION_MODEL, "gemini": GEMINI_MODEL}
    # MODIFIED: Prompt now requests properties for relationships.
    # Cần sửa: 
    # Ở INSTRUCTION 1: thêm vào yếu tố domain giáo dục
    prompt = f"""
    You are an expert information extraction system. Your task is to read the following text and extract a knowledge graph from it.
    The graph consists of entities (instances), their attributes (properties), and the relationships between them.
    The list of known classes is: {existing_classes}

    Source Text:
    "{text.strip()}"

    Extract the information and return it as a single JSON object with three keys: "classes", "instances", and "relationships".
    
    INSTRUCTIONS:
    1.  **JSON Structure**:
        - `classes`: A list of object types. You can add new classes to this list if appropriate. But avoid adding classes as much as possible when it could be an instance of an existing class.
        - `instances`: A list of objects. Each object must have an "id", "name", "type", and "properties".
            - `id`: Must be a unique, lowercase string with underscores instead of spaces and no accents (e.g., "dh_bach_khoa_tphcm").
            - `name`: The name of the entity as found in the text.
            - `type`: The class of the entity (e.g., "Trường đại học").
            - `properties`: A JSON object containing key-value attributes of the instance extracted *directly from the text*. If no properties are found, use an empty object {{}}. The keys for properties must be in snake_case.
        - `relationships`: A list of objects, each with "start_node_id", "type", "end_node_id", and "properties".
            - The relationship "type" must be in UPPER_SNAKE_CASE.
            - `properties`: A JSON object for attributes of the relationship itself (e.g., date, location, manner). If none, use an empty object {{}}.

    2.  **Language Requirement**: The *content* of the output (names, types, definitions, property values) MUST be in Vietnamese. The JSON keys ("id", "name", "properties", etc.) must remain in English as specified.

    3.  **Domain Restriction - Education only**: 
        - Extract a class only if it represents entities used in formal education ecosystems (K-12, higher-ed, vocational, research).

        - In-scope categories (examples): Institution (University, Faculty, Department, School), Program (DegreeProgram, Major, Minor), Course (Course, CourseSection, Module, Syllabus, Curriculum), People with roles (Student, Instructor, TeachingAssistant, Advisor, Alumni, Staff), Assessment (Exam, Quiz, Assignment, Project, Grade, Credit, GPA), Enrollment & Admin (Enrollment, Prerequisite, Semester/Term, Timetable, ClassSchedule, Scholarship, TuitionInvoice), Learning Activities (Lecture, LaboratorySession, Tutorial, Seminar, Internship, Thesis), Infrastructure (Classroom, LabRoom, Campus, Library), Research (ResearchGroup, Publication, Supervision) when directly tied to teaching/research contexts in the input.

        - Generic terms rule: Generic classes (Person, Organization, Event, Location, Document) are not allowed unless specialized with an explicit education role or qualifier (e.g., Student, UniversityDepartment, ExamEvent, CampusLocation, CourseSyllabus).

        - Out-of-scope (hard exclude): consumer products, finance/crypto, restaurants, pets, celebrities, generic commerce items, weather, unrelated medical records, movies/entertainment (unless explicitly course material), vehicles, generic “Account/Ticket/Invoice” (unless TuitionInvoice), generic City/Country (unless used as CampusLocation).

        - Edge cases:
            - Publication is in-scope only if academic (e.g., course reading or thesis).
            - Room/Facility is in-scope only if instructional (Classroom/LabRoom/Library).
            - “Person” must be emitted as a specialized educational role (Student/Instructor/etc.), not as a bare Person.

    4.  **Example JSON Format**:
        {{
          "classes": ["Trường đại học", "Quốc gia", "Thành phố"],
          "instances": [
            {{ "id": "dh_bach_khoa_tphcm", "name": "Trường Đại học Bách khoa", "type": "Trường đại học", "properties": {{ "nam_thanh_lap": 1957, "ten_goi_khac": "ĐH Bách Khoa TPHCM" }} }},
            {{ "id": "viet_nam", "name": "Việt Nam", "type": "Quốc gia", "properties": {{ "thu_do": "Hà Nội" }} }}
          ],
          "relationships": [
            {{ "start_node_id": "dh_bach_khoa_tphcm", "type": "TRU_SO_TAI", "end_node_id": "ha_noi", "properties": {{ "tu_nam": 1957 }} }}
          ]
        }}

    IMPORTANT: Only extract properties that are explicitly mentioned in the source text. Do not infer or add information that is not present.

    Extracted JSON object:
    """
    response_text = call_llm(prompt, LLM_PROVIDER, model_map[LLM_PROVIDER])
    if not response_text: return {"instances": [], "relationships": [], "classes": []}
    try:
        extracted_data = json.loads(clean_llm_json_output(response_text))
        if 'instances' not in extracted_data: extracted_data['instances'] = []
        if 'relationships' not in extracted_data: extracted_data['relationships'] = []
        if 'classes' not in extracted_data: extracted_data['classes'] = []
        return extracted_data
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM extraction JSON: {e}\nRaw response was: {response_text}")
        return {"instances": [], "relationships": [], "classes": []}


def reject_classes_that_are_instances(open_graph: Dict, existing_classes: List[str]) -> Tuple[Dict, List[Dict]]:
    """
    Checks if any newly extracted 'classes' are actually instances of existing classes.
    If so, it modifies the open_graph to correct this and logs the change, safely merging properties.
    """
    print("\n--- Stage 2: Class Rejection (Checking for misclassified instances) ---")
    rejection_log = []
    if not existing_classes or not open_graph.get('instances'):
        return open_graph, rejection_log

    model_map = {"openai": OPENAI_REJECTION_MODEL, "gemini": GEMINI_MODEL}
    modified_graph = copy.deepcopy(open_graph)
    newly_found_types = set(inst['type'] for inst in modified_graph.get('instances', []))
    new_potential_classes = list(newly_found_types - set(existing_classes))

    if not new_potential_classes:
        return open_graph, rejection_log

    classes_to_remove_as_type = set()

    for new_class_name in new_potential_classes:
        prompt = f"""
        You are a knowledge graph ontology expert. Your task is to determine if a given term, which was mistakenly identified as a "Class", is actually an "Instance" of an existing class.

        Term to check: "{new_class_name}"

        List of existing, valid Classes:
        {existing_classes}

        Is the term "{new_class_name}" an instance of one of the classes in the list above?
        - If YES, respond with the single most appropriate class name from the list.
        - If NO, or if you are unsure, respond with the exact word "NONE".

        Do not provide any explanation or other text. Your response should be ONLY the class name or "NONE".
        """  # Unchanged prompt
        response = call_llm(prompt, LLM_PROVIDER, model_map[LLM_PROVIDER], max_tokens=50)
        parent_class_name = response.strip()

        if parent_class_name != "NONE" and parent_class_name in existing_classes:
            print(f"  - Class Rejection: LLM identified '{new_class_name}' as an instance of '{parent_class_name}'.")
            classes_to_remove_as_type.add(new_class_name)
            new_instance_id = _create_id_from_name(new_class_name)

            aggregated_properties = {}
            ids_to_replace = []
            original_instances = []
            for inst in modified_graph['instances']:
                if inst['type'] == new_class_name:
                    ids_to_replace.append(inst['id'])
                    original_instances.append(copy.deepcopy(inst))
                    merge_properties_uniquely(aggregated_properties, inst.get('properties', {}))

            new_instance = {
                "id": new_instance_id,
                "name": new_class_name,
                "type": parent_class_name,
                "properties": aggregated_properties
            }

            rejection_log.append({
                "rejected_class": new_class_name, "corrected_as_instance_of": parent_class_name,
                "created_new_instance": new_instance, "original_misclassified_instances_removed": original_instances
            })
            modified_graph['instances'] = [inst for inst in modified_graph['instances'] if
                                           inst['type'] != new_class_name]
            modified_graph['instances'].append(new_instance)

            for rel in modified_graph.get('relationships', []):
                if rel['start_node_id'] in ids_to_replace: rel['start_node_id'] = new_instance_id
                if rel['end_node_id'] in ids_to_replace: rel['end_node_id'] = new_instance_id

    if 'classes' in modified_graph:
        modified_graph['classes'] = [c for c in modified_graph['classes'] if c not in classes_to_remove_as_type]
    return modified_graph, rejection_log


def define_new_schemas(open_graph: Dict, existing_graph: Dict) -> Dict:
    """Uses an LLM to generate definitions for new, unseen classes and relationship types."""
    print("--- Stage 3: Defining New Schema Elements ---")
    existing_classes = set(c for c in existing_graph.get('classes', []))
    new_classes = set(inst['type'] for inst in open_graph.get('instances', [])) - existing_classes
    existing_rel_types = set(rt['name'] for rt in existing_graph.get('relationship_types', []))
    new_rel_types = set(rel['type'] for rel in open_graph.get('relationships', [])) - existing_rel_types

    if not new_classes and not new_rel_types:
        return {}

    model_map = {"openai": OPENAI_DEFINITION_MODEL, "gemini": GEMINI_MODEL}
    prompt = f"""You are a knowledge modeler. Your task is to provide concise, general definitions for the following new Classes and Relationship Types.

    - New Classes to define: {list(new_classes) if new_classes else "None"}
    - New Relationship Types to define: {list(new_rel_types) if new_rel_types else "None"}

    Return the result as a single JSON object with two keys: "class_definitions" and "relationship_definitions".
    - Each item in the definition lists must be an object with "type" and "definition".

    **Language Requirement**: All values for "type" and "definition" in your response MUST be in **Vietnamese**.

    Example JSON Format:
    {{
      "class_definitions": [
        {{"type": "Ca sĩ", "definition": "Một người biểu diễn âm nhạc bằng giọng hát của họ."}}
      ],
      "relationship_definitions": [
        {{"type": "SANG_TAC_BOI", "definition": "Chỉ mối quan hệ mà chủ thể được tạo ra bởi đối tượng."}}
      ]
    }}

    Definitions JSON object:"""  # Unchanged prompt
    response_text = call_llm(prompt, LLM_PROVIDER, model_map[LLM_PROVIDER])
    if not response_text: return {"class_definitions": [], "relationship_definitions": []}
    try:
        definitions = json.loads(clean_llm_json_output(response_text))
        print("LLM Generated Definitions:", json.dumps(definitions, indent=2, ensure_ascii=False))
        return definitions
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM definition JSON: {e}\nRaw response was: {response_text}")
        return {"class_definitions": [], "relationship_definitions": []}


# --- FULLY REVISED AND OPTIMIZED CANONICALIZER CLASS ---
# Cần sửa: các phần 4.
class GraphCanonicalizer:
    def __init__(self, initial_graph: Dict):
        print("--- Initializing Graph Canonicalizer ---")
        self.graph = copy.deepcopy(initial_graph)
        self.provider = LLM_PROVIDER
        self.verification_model = OPENAI_VERIFICATION_MODEL if self.provider == "openai" else GEMINI_MODEL
        self.AUTO_MERGE_SIMILARITY_THRESHOLD = 0.95

        try:
            self.embedder = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device='cuda')
            print("SentenceTransformer loaded on CUDA.")
        except Exception:
            print("CUDA not available. Falling back to CPU.")
            self.embedder = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device='cpu')

        self._initialize_internal_state()
        print("Initialization complete.\n")

    def _initialize_internal_state(self):
        """Initializes the state from the full graph. Called only once."""
        self.class_defs = {item['type']: item['definition'] for item in self.graph.get('class_definitions', [])}
        self.rel_defs = {item['type']: item['definition'] for item in self.graph.get('relationship_definitions', [])}
        self.class_names = list(self.class_defs.keys())
        self.rel_names = list(self.rel_defs.keys())

        if self.class_names:
            self.class_embeddings = self.embedder.encode(list(self.class_defs.values()), show_progress_bar=True)
        else:
            self.class_embeddings = np.array([])
        if self.rel_names:
            self.rel_embeddings = self.embedder.encode(list(self.rel_defs.values()), show_progress_bar=True)
        else:
            self.rel_embeddings = np.array([])

        self.instance_details = {inst['id']: inst for inst in self.graph.get('instances', [])}
        self.instance_embeddings_by_type = {}

        instances_by_type = {}
        for inst in self.graph.get('instances', []):
            type_key = inst.get('type')
            if type_key not in instances_by_type: instances_by_type[type_key] = []
            instances_by_type[type_key].append(inst)

        for type_key, instances in tqdm.tqdm(instances_by_type.items(), desc="Encoding Instances"):
            names = [inst['name'] for inst in instances]
            ids = [inst['id'] for inst in instances]
            if names:
                self.instance_embeddings_by_type[type_key] = {
                    'ids': ids, 'embeddings': self.embedder.encode(names, show_progress_bar=False)
                }

        # NEW: Lookup for existing relationships to merge properties
        self.rel_lookup = {f"{rel['start_node_id']}|{rel['type']}|{rel['end_node_id']}": rel for rel in
                           self.graph.get('relationships', [])}

    def _find_best_candidate(self, query_text: str, target_embeddings: np.ndarray, target_ids: List[str],
                             threshold: float) -> Optional[Tuple[str, float]]:
        if not target_ids or target_embeddings.size == 0: return None
        query_emb = self.embedder.encode(query_text, show_progress_bar=False)
        scores = np.dot(target_embeddings, query_emb.T) / (
                    np.linalg.norm(target_embeddings, axis=1) * np.linalg.norm(query_emb))
        best_match_idx = np.argmax(scores)
        if (best_score := scores[best_match_idx]) > threshold:
            best_candidate_id = target_ids[best_match_idx]
            print(
                f"    - Top candidate for '{query_text[:30]}...' is '{best_candidate_id}' with similarity score: {best_score:.4f}")
            return best_candidate_id, best_score
        return None

    def _verify_mappings_in_batch_with_llm(self, verification_requests: List[Dict]) -> List[bool]:
        """
        Verifies a batch of potential mappings with a single LLM call.
        """
        if not verification_requests:
            return []

        print(f"\n--- Sending a batch of {len(verification_requests)} items to LLM for verification ---")
        prompt_items = []

        # --- CORRECTED LOGIC ---
        for i, req in enumerate(verification_requests):
            if req['type'] == 'schema':
                # Safely escape quotes inside JSON strings
                open_def_escaped = json.dumps(req['open_def'])
                candidate_def_escaped = json.dumps(req['candidate_def'])
                item_str = f'{{"index": {i}, "item_type": "{req["element_category"]}", "new_item": {{"name": "{req["open_type"]}", "definition": {open_def_escaped}}}, "existing_item": {{"name": "{req["candidate_type"]}", "definition": {candidate_def_escaped}}} }}'

            elif req['type'] == 'instance':
                # Use json.dumps for the whole object to handle complex properties
                new_instance_json = json.dumps(req["new_instance"], ensure_ascii=False)
                candidate_instance_json = json.dumps(req["candidate_instance"], ensure_ascii=False)
                item_str = f'{{"index": {i}, "item_type": "Instance", "new_item": {new_instance_json}, "existing_item": {candidate_instance_json} }}'

            else:
                # Skip unknown request types
                continue

            prompt_items.append(item_str)
        # --- END OF CORRECTION ---

        if not prompt_items:
            return []

        prompt = f"""You are a strict data modeling and entity resolution expert. I will provide you with a JSON list of items. Your task is to determine if the "new_item" is semantically equivalent to the "existing_item" and should be merged.
        - For "Class" or "Relationship Type", check if their definitions mean the same thing.
        - For "Instance", check if they refer to the exact same real-world entity.

        Evaluate each item in the list below:
        [{', '.join(prompt_items)}]

        Respond with a single JSON object with a single key "decisions", which is a list of objects. Each object must have an "index" and a boolean "merge_decision".
        Example: {{"decisions": [{{"index": 0, "merge_decision": true}}, {{"index": 1, "merge_decision": false}}]}}
        Your JSON response:"""

        response_text = call_llm(prompt, self.provider, self.verification_model)
        try:
            results = json.loads(clean_llm_json_output(response_text))
            # Handle cases where the LLM might not return a decision for every item
            decision_map = {item['index']: item['merge_decision'] for item in results.get('decisions', [])}
            # Return decisions in the original order, defaulting to False if missing
            return [decision_map.get(i, False) for i in range(len(verification_requests))]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(
                f"!! CRITICAL: Failed to parse batched LLM verification response. Error: {e}\nRaw response: {response_text}")
            return [False] * len(verification_requests)
    
    def _incrementally_update_state(self, new_schema_defs: List = [], new_instances: List = []):
        """Incrementally updates the internal state with new items without full re-encoding."""
        # Update Schema
        if new_classes := [d for d in new_schema_defs if d['__type__'] == 'class']:
            new_embeddings = self.embedder.encode([d['definition'] for d in new_classes], show_progress_bar=False)
            self.class_names.extend([d['type'] for d in new_classes])
            for d in new_classes: self.class_defs[d['type']] = d['definition']
            self.class_embeddings = np.vstack(
                [self.class_embeddings, new_embeddings]) if self.class_embeddings.size else new_embeddings
        if new_rels := [d for d in new_schema_defs if d['__type__'] == 'relationship']:
            new_embeddings = self.embedder.encode([d['definition'] for d in new_rels], show_progress_bar=False)
            self.rel_names.extend([d['type'] for d in new_rels])
            for d in new_rels: self.rel_defs[d['type']] = d['definition']
            self.rel_embeddings = np.vstack(
                [self.rel_embeddings, new_embeddings]) if self.rel_embeddings.size else new_embeddings
        # Update Instances
        if new_instances:
            new_instances_by_type = {}
            for inst in new_instances:
                if (t := inst['type']) not in new_instances_by_type: new_instances_by_type[t] = []
                new_instances_by_type[t].append(inst)

            for inst_type, instances in new_instances_by_type.items():
                new_embeddings = self.embedder.encode([i['name'] for i in instances], show_progress_bar=False)
                if inst_type in self.instance_embeddings_by_type:
                    self.instance_embeddings_by_type[inst_type]['ids'].extend([i['id'] for i in instances])
                    self.instance_embeddings_by_type[inst_type]['embeddings'] = np.vstack(
                        [self.instance_embeddings_by_type[inst_type]['embeddings'], new_embeddings])
                else:
                    self.instance_embeddings_by_type[inst_type] = {'ids': [i['id'] for i in instances],
                                                                   'embeddings': new_embeddings}
                for inst in instances: self.instance_details[inst['id']] = inst

    def canonicalize_and_enrich(self, open_graph: Dict, new_definitions: Dict) -> Dict:
        print("\n--- Stage 4: Canonicalizing and Enriching Graph (with Batched LLM Verification) ---")
        change_log = {"new_classes": [], "merged_classes": [], "new_rels": [], "merged_rels": [], "new_instances": [],
                      "merged_instances": []}

        # === 4.1 BATCH PROCESS SCHEMA ===
        class_map, rel_map = {}, {}
        schema_reqs, pending_schema = [], []
        for d in new_definitions.get('class_definitions', []):
            if res := self._find_best_candidate(d['definition'], self.class_embeddings, self.class_names,
                                                SCHEMA_SIMILARITY_THRESHOLD):
                cand, score = res
                if score >= self.AUTO_MERGE_SIMILARITY_THRESHOLD:
                    class_map[d['type']] = cand; change_log['merged_classes'].append((d['type'], cand))
                else:
                    schema_reqs.append({'type': 'schema', 'element_category': 'Class', 'open_type': d['type'],
                                        'open_def': d['definition'], 'candidate_type': cand,
                                        'candidate_def': self.class_defs[cand], '__original_def__': d})
            else:
                d['__type__'] = 'class'; pending_schema.append(d); class_map[d['type']] = d['type']
        for d in new_definitions.get('relationship_definitions', []):
            if res := self._find_best_candidate(d['definition'], self.rel_embeddings, self.rel_names,
                                                SCHEMA_SIMILARITY_THRESHOLD):
                cand, score = res
                if score >= self.AUTO_MERGE_SIMILARITY_THRESHOLD:
                    rel_map[d['type']] = cand; change_log['merged_rels'].append((d['type'], cand))
                else:
                    schema_reqs.append(
                        {'type': 'schema', 'element_category': 'Relationship Type', 'open_type': d['type'],
                         'open_def': d['definition'], 'candidate_type': cand, 'candidate_def': self.rel_defs[cand],
                         '__original_def__': d})
            else:
                d['__type__'] = 'relationship'; pending_schema.append(d); rel_map[d['type']] = d['type']

        for req, verified in zip(schema_reqs, self._verify_mappings_in_batch_with_llm(schema_reqs)):
            if verified:
                if req['element_category'] == 'Class':
                    class_map[req['open_type']] = req['candidate_type']; change_log['merged_classes'].append(
                        (req['open_type'], req['candidate_type']))
                else:
                    rel_map[req['open_type']] = req['candidate_type']; change_log['merged_rels'].append(
                        (req['open_type'], req['candidate_type']))
            else:
                d = req['__original_def__'];
                d['__type__'] = 'class' if req['element_category'] == 'Class' else 'relationship';
                pending_schema.append(d)
                if req['element_category'] == 'Class':
                    class_map[req['open_type']] = req['open_type']
                else:
                    rel_map[req['open_type']] = req['open_type']

        if pending_schema:
            for d in pending_schema:
                if d['__type__'] == 'class':
                    self.graph['classes'].append(d['type']); self.graph['class_definitions'].append(d); change_log[
                        'new_classes'].append(d)
                else:
                    self.graph['relationship_types'].append({"name": d['type']}); self.graph[
                        'relationship_definitions'].append(d); change_log['new_rels'].append(d)
            self._incrementally_update_state(new_schema_defs=pending_schema)

        # === 4.2 BATCH PROCESS INSTANCES ===
        id_map, to_add, inst_reqs = {}, [], []
        for inst in open_graph.get('instances', []):
            inst['type'] = class_map.get(inst['type'], inst['type'])
            if (t_data := self.instance_embeddings_by_type.get(inst['type'])) and (
            res := self._find_best_candidate(inst['name'], t_data['embeddings'], t_data['ids'],
                                             INSTANCE_SIMILARITY_THRESHOLD)):
                cand_id, score = res
                cand_inst = self.instance_details[cand_id]
                if score >= self.AUTO_MERGE_SIMILARITY_THRESHOLD:
                    id_map[inst['id']] = cand_id;
                    change_log['merged_instances'].append((inst, cand_inst))
                    merge_properties_uniquely(self.instance_details[cand_id].setdefault('properties', {}),
                                              inst.get('properties'))
                else:
                    inst_reqs.append({'type': 'instance', 'new_instance': inst, 'candidate_instance': cand_inst})
            else:
                to_add.append(inst); id_map[inst['id']] = inst['id']

        for req, verified in zip(inst_reqs, self._verify_mappings_in_batch_with_llm(inst_reqs)):
            if verified:
                id_map[req['new_instance']['id']] = req['candidate_instance']['id'];
                change_log['merged_instances'].append((req['new_instance'], req['candidate_instance']))
                merge_properties_uniquely(
                    self.instance_details[req['candidate_instance']['id']].setdefault('properties', {}),
                    req['new_instance'].get('properties'))
            else:
                to_add.append(req['new_instance']); id_map[req['new_instance']['id']] = req['new_instance']['id']

        if to_add:
            self.graph['instances'].extend(to_add);
            change_log['new_instances'].extend(to_add)
            self._incrementally_update_state(new_instances=to_add)

        # === 4.3 PROCESS RELATIONSHIPS ===
        print("\n--- 4.3: Processing & Merging Relationships ---")
        for rel in open_graph.get('relationships', []):
            rel['type'] = rel_map.get(rel['type'], rel['type'])
            rel['start_node_id'] = id_map.get(rel['start_node_id'], rel['start_node_id'])
            rel['end_node_id'] = id_map.get(rel['end_node_id'], rel['end_node_id'])
            if rel['start_node_id'] == rel['end_node_id']: continue

            rel_key = f"{rel['start_node_id']}|{rel['type']}|{rel['end_node_id']}"
            if existing_rel := self.rel_lookup.get(rel_key):
                print(f"  - Merging properties for existing relationship: {rel_key}")
                merge_properties_uniquely(existing_rel.setdefault('properties', {}), rel.get('properties'))
            else:
                print(f"  + Adding new relationship: {rel_key}")
                self.graph['relationships'].append(rel)
                self.rel_lookup[rel_key] = rel

        return change_log


def setup_folders(input_dir: str, logs_dir: str, prompts_dir: str):
    """Checks for input and log folders and creates them."""
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        with open(os.path.join(input_dir, "sample_input.txt"), "w", encoding="utf-8") as f:
            f.write(
                "Trường Đại học Bách khoa, thành lập năm 1957, là một trường đại học kỹ thuật hàng đầu tại Việt Nam.")
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)
    if not os.path.exists(prompts_dir): os.makedirs(prompts_dir)


if __name__ == "__main__":
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"--- Starting Run: {run_timestamp} ---")

    LOGS_FOLDER = f"{LOGS_FOLDER_BASE}_{run_timestamp}"
    PROMPTS_FOLDER = f"{PROMPTS_FOLDER_BASE}_{run_timestamp}"
    setup_folders(INPUT_FOLDER, LOGS_FOLDER, PROMPTS_FOLDER)

    input_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "**", "*.txt"), recursive=True))
    if not input_files:
        print(f"No '.txt' files found in '{INPUT_FOLDER}'. Please add text files to process.")
        exit()

    print(f"\nFound {len(input_files)} text file(s) to process.")
    canonicalizer = GraphCanonicalizer(INITIAL_KNOWLEDGE_GRAPH)
    processing_order = []

    for file_path in tqdm.tqdm(input_files, desc="Processing files"):
        print(f"\n\n{'=' * 20} Processing: {file_path} {'=' * 20}")
        CURRENT_FILE_LLM_LOGS.clear()
        processing_order.append(os.path.relpath(file_path, INPUT_FOLDER))

        log_data = {}
        file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        processing_log_path = os.path.join(LOGS_FOLDER, f"log_{file_name_no_ext}.json")
        prompts_log_path = os.path.join(PROMPTS_FOLDER, f"prompts_{file_name_no_ext}.json")

        with open(file_path, "r", encoding="utf-8") as f:
            input_text = f.read()
        if not input_text.strip(): continue

        open_graph, rejection_log = (
            extract_graph_from_text(input_text, canonicalizer.graph['classes']),
            canonicalizer.graph['classes']
        )
        log_data['rejected_classes_as_instances'] = rejection_log
        if not open_graph.get('instances'): continue

        new_definitions = define_new_schemas(open_graph, canonicalizer.graph)
        change_log = canonicalizer.canonicalize_and_enrich(open_graph, new_definitions)
        log_data.update(change_log)

        with open(processing_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"\n  -> Processing log saved to '{processing_log_path}'")

        with open(prompts_log_path, "w", encoding="utf-8") as f:
            json.dump(CURRENT_FILE_LLM_LOGS, f, indent=2, ensure_ascii=False)
        print(f"  -> LLM prompts and answers saved to '{prompts_log_path}'")

    # --- Save Final Graph ---
    base, ext = os.path.splitext(FINAL_KG_FILE)
    timestamped_final_kg_file = f"{base}_{run_timestamp}{ext}"

    print("\n\n--- SAVING FINAL KNOWLEDGE GRAPH ---")
    with open(timestamped_final_kg_file, "w", encoding="utf-8") as f:
        json.dump(canonicalizer.graph, f, indent=2, ensure_ascii=False)
    print(f"\nProcessing complete. Final graph saved to '{timestamped_final_kg_file}'")

    order_log_path = os.path.join(LOGS_FOLDER, "processing_order.json")
    with open(order_log_path, "w", encoding="utf-8") as f:
        json.dump({"processing_order": processing_order}, f, indent=2)
    print(f"Processing order saved to '{order_log_path}'")