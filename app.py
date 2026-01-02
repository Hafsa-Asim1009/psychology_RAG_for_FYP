# ==========================================================
# ENHANCED PSYCHOLOGY RAG - COUNSELOR ROLE-PLAY VERSION (UPDATED)
# ==========================================================

# 1. UPDATED INSTALLATION
#!pip uninstall google-generativeai -y -q
#!pip install -q -U google-genai
#!pip install -q pinecone-client sentence-transformers tqdm python-dotenv
#!pip install -q "pinecone[grpc]"

print("✅ Updated dependencies installed!")

# ==========================================================
# 2. MAIN CODE - COUNSELOR ROLE-PLAY EDITION (WITH FLOW CONTROL)
# ==========================================================

import os
import re
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Vector Database
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Embeddings
from sentence_transformers import SentenceTransformer

# Modern Gemini API
from google import genai

print("=" * 70)
print("🧠 ENHANCED PSYCHOLOGY RAG - COUNSELOR SIMULATION")
print("✅ Role-Playing Counselor | FYP Ready | WITH FLOW CONTROL")
print("=" * 70)

# ==========================================================
# ENUMS & CONSTANTS
# ==========================================================

class EmotionState(Enum):
    DEPRESSED = "depressed"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    STRESSED = "stressed"
    CALM = "calm"
    HAPPY = "happy"
    NEUTRAL = "neutral"
    AGITATED = "agitated"

class CounselingStyle(Enum):
    EMPATHIC = "empathic"
    PERSONALITY = "personality"
    CRISIS = "crisis"
    GENERAL = "general"
    THERAPEUTIC = "therapeutic"
    COUNSELOR = "counselor"  # New style for counseling sessions

class ResponseTone(Enum):
    EMPATHETIC = "empathetic"
    INFORMATIVE = "informative"
    PRACTICAL = "practical"
    CRISIS = "crisis"
    SUPPORTIVE = "supportive"
    COUNSELING = "counseling"

# NEW: Counseling Phase Tracking
class CounselingPhase(Enum):
    ASSESSMENT = "assessment"  # Initial questioning phase
    INTERVENTION = "intervention"  # Providing therapeutic techniques
    EXPLORATION = "exploration"  # Deeper questioning
    CLOSING = "closing"  # Wrapping up

# ==========================================================
# MAIN RAG CLASS - COUNSELOR ROLE-PLAY (UPDATED)
# ==========================================================

class EnhancedPsychologyRAG:
    def __init__(self, 
                 index_name="psychology-fyp",
                 use_gemini=True,
                 embedding_model="all-MiniLM-L6-v2",
                 gemini_model="gemini-2.5-flash-lite"):
        """
        Enhanced RAG system with counselor role-play and flow control
        """
        self.gemini_model = gemini_model
        self.load_secrets(use_gemini)
        self.initialize_components(embedding_model)
        self.setup_pinecone(index_name)
        
        # NEW: Counseling session tracking
        self.counseling_phase = CounselingPhase.ASSESSMENT
        self.assessment_count = 0
        self.max_assessment_questions = 3  # Maximum questions before intervention
        self.client_issue_type = None
        self.counseling_tools_used = []
        
        # Counseling configuration with role-play prompts
        self.counseling_prompts = self.create_counseling_prompts()
        self.safety_warnings = self.create_safety_warnings()
        self.therapy_exercises = self.create_therapy_exercises()
        self.counseling_questions = self.create_counseling_questions()
        
        # Session management
        self.session_history = []
        self.conversation_context = []
        self.user_progress = {}  # Track user's counseling progress
        
        # Rate Limiting
        self.rate_limit_config = {
            "requests_per_minute": 15,
            "requests_per_day": 500,
            "tokens_per_minute": 30000,
            "min_request_interval": 4.0,
        }
        self.rate_limit_state = {
            "request_timestamps": [],
            "daily_request_count": 0,
            "daily_token_count": 0,
            "last_reset_date": datetime.now().date()
        }
        
        # Statistics
        self.stats = {
            "queries_made": 0,
            "counseling_style": {},
            "response_tone": {},
            "rate_limit_hits": 0,
            "session_start": datetime.now().isoformat(),
            "counseling_sessions_started": 0,
            "counseling_phases": {}  # NEW: Track phases used
        }
        
        print(f"✅ RAG System Initialized as Counseling Assistant")
        print(f"   Model: {gemini_model}")
        print(f"   Rate Limit: {self.rate_limit_config['requests_per_minute']} RPM")
        print(f"   Flow Control: ACTIVE (max {self.max_assessment_questions} assessment questions)")
    
    # ========== CORE CONFIGURATION METHODS ==========
    
    def load_secrets(self, use_gemini):
        """Load API keys"""
        print("\n🔐 Loading API keys...")
        
        try:
            load_dotenv()
            
            # Pinecone API Key
            self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
            if not self.PINECONE_API_KEY:
                try:
                    from kaggle_secrets import UserSecretsClient
                    user_secrets = UserSecretsClient()
                    self.PINECONE_API_KEY = user_secrets.get_secret("PINECONE_API_KEY")
                except:
                    pass
            
            if not self.PINECONE_API_KEY:
                raise ValueError("❌ PINECONE_API_KEY not found!")
            print("   ✅ Pinecone API key loaded")
            
            # Gemini API Key
            if use_gemini:
                self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                if not self.GEMINI_API_KEY:
                    try:
                        from kaggle_secrets import UserSecretsClient
                        user_secrets = UserSecretsClient()
                        self.GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
                    except:
                        pass
                
                if not self.GEMINI_API_KEY:
                    print("   ⚠️ Gemini API key not found, using simple responses")
                    self.use_gemini = False
                else:
                    os.environ["GEMINI_API_KEY"] = self.GEMINI_API_KEY
                    self.use_gemini = True
                    print(f"   ✅ Gemini API key loaded")
            else:
                self.use_gemini = False
                
        except Exception as e:
            print(f"   ⚠️ Secrets loading issue: {e}")
            self.PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
            self.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
            self.use_gemini = bool(self.GEMINI_API_KEY) and use_gemini
    
    def initialize_components(self, embedding_model):
        """Initialize all components"""
        print("\n⚙️ Initializing components...")
        
        # Embedding model
        print(f"   Loading embedding model: {embedding_model}")
        try:
            self.embed_model = SentenceTransformer(embedding_model)
            self.embedding_dimension = self.embed_model.get_sentence_embedding_dimension()
            print(f"   ✅ Embedding model loaded (dimension: {self.embedding_dimension})")
        except Exception as e:
            print(f"   ❌ Failed to load embedding model: {e}")
            raise
        
        # Gemini client
        if self.use_gemini:
            print("   Initializing Gemini client...")
            self.gemini_client = self.initialize_gemini()
        else:
            self.gemini_client = None
        
        print("✅ Components initialized")
    
    def initialize_gemini(self):
        """Initialize Gemini client with correct models"""
        try:
            test_models = [
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash",
                "gemini-2.0-flash-lite",
            ]
            
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            
            for model_name in test_models:
                try:
                    test_prompt = "Please respond with 'Connection successful'"
                    response = client.models.generate_content(
                        model=model_name,
                        contents=test_prompt
                    )
                    
                    self.gemini_model = model_name
                    print(f"   ✅ Gemini connected with model: {model_name}")
                    return client
                    
                except Exception as model_error:
                    error_msg = str(model_error)
                    if "429" in error_msg:
                        print(f"   ⚠️ Model {model_name}: Rate limit hit")
                        time.sleep(5)
                        continue
                    elif "404" in error_msg:
                        print(f"   ⚠️ Model {model_name}: Not found")
                        continue
                    else:
                        print(f"   ⚠️ Model {model_name} failed: {error_msg[:80]}")
                        continue
            
            print("   ❌ All Gemini models failed, using simple responses")
            return None
            
        except Exception as e:
            print(f"   ❌ Gemini initialization error: {e}")
            return None
    
    def setup_pinecone(self, index_name):
        """Connect to Pinecone index"""
        print(f"\n🗄️ Connecting to Pinecone index: '{index_name}'")
        
        try:
            self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
            
            if index_name not in self.pc.list_indexes().names():
                print(f"   ⚠️ Index '{index_name}' not found")
                return None
            else:
                self.index = self.pc.Index(index_name)
                stats = self.index.describe_index_stats()
                print(f"   ✅ Connected to existing index")
                print(f"   📊 Vectors: {stats.get('total_vector_count', 0)}")
                
        except Exception as e:
            print(f"❌ Pinecone connection error: {e}")
            raise
    
    # ========== COUNSELOR ROLE-PLAY METHODS (UPDATED) ==========
    
    def create_counseling_prompts(self):
        """Counselor role-play prompts - WITH FLOW CONTROL"""
        return {
            CounselingStyle.EMPATHIC.value: """You are an empathetic mental health counselor conducting a therapy session. You are talking with a client who needs emotional support.

CLIENT'S CURRENT STATE:
- Emotion: {emotion}
- Query: "{query}"
- Previous context: {conversation_context}

CURRENT COUNSELING PHASE: {counseling_phase}

RELEVANT PSYCHOLOGY KNOWLEDGE:
{context}

AS A COUNSELOR, RESPOND BY:
1. Starting with VALIDATION: "I hear that you're feeling..." or "That sounds really challenging"
2. Showing EMPATHY and UNDERSTANDING
3. If in ASSESSMENT phase: Ask ONLY 1-2 clarifying questions MAXIMUM
4. If in INTERVENTION phase: Provide practical techniques/exercises IMMEDIATELY
5. If client asks for CURE/SOLUTION: Move directly to intervention
6. Maintaining a WARM, PROFESSIONAL tone

{phase_specific_instructions}

COUNSELOR'S RESPONSE:""",

            CounselingStyle.COUNSELOR.value: """You are a professional mental health counselor conducting a therapy session. Your client has shared their issue and is ready for therapeutic intervention.

CLIENT'S ISSUE SUMMARY:
- Main concern: {query}
- Reported symptoms: {symptoms_summary}
- Duration: {issue_duration}
- Impact: {issue_impact}

CURRENT COUNSELING PHASE: {counseling_phase}

RELEVANT THERAPEUTIC KNOWLEDGE:
{context}

AVAILABLE THERAPY EXERCISES:
{therapy_exercises}

GUIDELINES FOR CURRENT PHASE:

ASSESSMENT PHASE (when information is incomplete):
1. Ask 1-2 clarifying questions MAXIMUM
2. Focus on understanding the core issue
3. Then move to intervention

INTERVENTION PHASE (when issue is clear):
1. PROVIDE IMMEDIATE THERAPEUTIC TECHNIQUES
2. Use evidence-based approaches
3. Give practical steps/exercises
4. Explain why it works
5. Ask for commitment to try

EXPLORATION PHASE (after initial intervention):
1. Ask about thoughts/feelings related to the issue
2. Explore underlying patterns
3. Provide psychoeducation

{phase_specific_instructions}

IMPORTANT: DO NOT ASK MORE THAN 2 QUESTIONS IN A ROW. After gathering basic information, MOVE TO INTERVENTION.

YOUR RESPONSE AS COUNSELOR:""",

            CounselingStyle.THERAPEUTIC.value: """You are a therapeutic counselor specializing in practical mental health techniques.

CLIENT'S REQUEST:
- Emotion: {emotion}
- Specific need: "{query}"

AVAILABLE THERAPY EXERCISES:
{therapy_exercises}

RELEVANT PSYCHOLOGY CONTEXT:
{context}

AS A THERAPEUTIC COUNSELOR:
1. ACKNOWLEDGE their need for practical help
2. SELECT appropriate exercises based on their emotion
3. PROVIDE CLEAR, STEP-BY-STEP instructions
4. EXPLAIN the psychological benefits
5. CHECK for understanding and comfort level
6. OFFER FOLLOW-UP support

THERAPEUTIC RESPONSE:""",

            CounselingStyle.PERSONALITY.value: """You are a counseling psychologist helping a client understand personality concepts.

CLIENT'S INTEREST:
- Emotion: {emotion}
- Question: "{query}"

RELEVANT PSYCHOLOGY CONTEXT:
{context}

AS A COUNSELING PSYCHOLOGIST:
1. Start with ENGAGEMENT: "That's an interesting area to explore..."
2. Provide PSYCHOEDUCATION about personality
3. CONNECT concepts to their personal experience
4. Ask SELF-REFLECTION questions
5. Suggest PRACTICAL APPLICATIONS
6. Maintain a SUPPORTIVE, EDUCATIONAL tone

COUNSELING PSYCHOLOGIST'S RESPONSE:""",

            CounselingStyle.CRISIS.value: """URGENT: You are a crisis counselor responding to a client in distress.

CLIENT IN CRISIS:
- Emotion: {emotion}
- Statement: "{query}"

CRISIS COUNSELING PROTOCOL:
1. IMMEDIATE VALIDATION: "This sounds incredibly painful and serious"
2. ASSESS SAFETY: "Your safety is my main concern right now"
3. PROVIDE CRISIS RESOURCES immediately
4. USE CALM, DIRECT language
5. OFFER CONTINUED SUPPORT
6. EMPHASIZE URGENT HELP AVAILABILITY

CRISIS RESOURCES:
- National Suicide Prevention Lifeline: 1-800-273-8255
- Crisis Text Line: Text HOME to 741741
- Emergency services: 911 or local number

CRISIS COUNSELOR RESPONSE:""",

            CounselingStyle.GENERAL.value: """You are a supportive mental health counselor. Client has: "{query}"

CONVERSATION HISTORY:
{conversation_context}

CURRENT COUNSELING PHASE: {counseling_phase}

RELEVANT PSYCHOLOGY INFORMATION:
{context}

AS A COUNSELOR, FOLLOW THIS FLOW:
1. If this is the FIRST mention of an issue: Ask 1-2 assessment questions
2. If issue is ALREADY DESCRIBED: Provide therapeutic intervention
3. If client asks for CURE/SOLUTION: Provide immediate techniques
4. Always include PRACTICAL STEPS they can try now

{phase_specific_instructions}

COUNSELOR'S RESPONSE:"""
        }
    
    def create_counseling_questions(self):
        """Questions for counseling sessions"""
        return {
            "opening_questions": [
                "What brings you here today?",
                "How have you been feeling lately?",
                "What's been on your mind recently?",
                "Is there something specific you'd like to talk about?"
            ],
            "exploration_questions": [
                "Can you tell me more about that feeling?",
                "How long have you been feeling this way?",
                "What happens when you feel this emotion?",
                "How does this affect your daily life?",
                "What have you tried so far to cope with this?"
            ],
            "reflective_questions": [
                "It sounds like you're feeling..., is that right?",
                "What I'm hearing is..., does that resonate with you?",
                "Help me understand what that experience is like for you"
            ],
            "goal_questions": [
                "What would you like to be different?",
                "How would you know things are getting better?",
                "What's one small step you could take?"
            ]
        }
    
    def create_safety_warnings(self):
        """Safety warnings for counselor role-play"""
        return {
            "low": "\n\n💡 **Counselor's Note**: Remember that I'm here to provide supportive conversation based on psychology principles. For ongoing personal concerns, consider connecting with a licensed mental health professional.",
            "medium": "\n\n⚠️ **Counselor's Note**: If these feelings persist or significantly impact your daily life, I encourage you to seek support from a qualified therapist or counselor who can provide ongoing care.",
            "high": "\n\n🚨 **URGENT COUNSELOR NOTE**: If you're having thoughts of harming yourself or others, please reach out to crisis services immediately:\n• National Suicide Prevention Lifeline: 1-800-273-8255\n• Crisis Text Line: Text HOME to 741741\n• Emergency: 911\n• You are not alone - immediate help is available"
        }
    
    def create_therapy_exercises(self):
        """Therapy exercises database"""
        return {
            "anxiety": [
                {
                    "name": "5-4-3-2-1 Grounding Technique",
                    "steps": [
                        "Look around and name 5 things you can SEE",
                        "Notice 4 things you can FEEL (textures, temperature)",
                        "Identify 3 things you can HEAR",
                        "Notice 2 things you can SMELL",
                        "Name 1 thing you can TASTE"
                    ],
                    "duration": "2-3 minutes",
                    "purpose": "Reduces anxiety by connecting to present moment"
                },
                {
                    "name": "Diaphragmatic Breathing",
                    "steps": [
                        "Sit comfortably with one hand on chest, one on belly",
                        "Breathe in slowly through nose for 4 seconds",
                        "Feel belly expand (chest should move less)",
                        "Hold breath for 2 seconds",
                        "Exhale slowly through mouth for 6 seconds",
                        "Repeat 5-10 times"
                    ],
                    "duration": "3-5 minutes",
                    "purpose": "Activates parasympathetic nervous system to calm anxiety"
                },
                {
                    "name": "Social Anxiety Exposure Ladder",
                    "steps": [
                        "List social situations from easiest to hardest",
                        "Start with the easiest (e.g., making eye contact with 1 person)",
                        "Practice for 5 minutes, notice your anxiety level",
                        "Gradually move to harder situations",
                        "Reward yourself after each step"
                    ],
                    "duration": "Daily, 10-15 minutes",
                    "purpose": "Gradually builds tolerance to social situations"
                }
            ],
            "depression": [
                {
                    "name": "Behavioral Activation: Small Wins",
                    "steps": [
                        "List 3 VERY small activities (e.g., make bed, drink water)",
                        "Schedule them at specific times",
                        "Complete one, then acknowledge the achievement",
                        "Gradually increase activity difficulty"
                    ],
                    "duration": "Throughout day",
                    "purpose": "Builds momentum and counters inactivity spiral"
                }
            ],
            "stress": [
                {
                    "name": "Progressive Muscle Relaxation",
                    "steps": [
                        "Start with feet: Tense muscles for 5 seconds",
                        "Release completely, notice the difference",
                        "Move upward: calves, thighs, abdomen, etc.",
                        "Finish with face and shoulders"
                    ],
                    "duration": "10 minutes",
                    "purpose": "Reduces physical tension from stress"
                }
            ]
        }
    
    # ========== FLOW CONTROL METHODS ==========
    
    def determine_counseling_phase(self, query, emotion):
        """Intelligently determine counseling phase based on conversation"""
        query_lower = query.lower()
        
        # Client is asking for solution/cure - IMMEDIATE INTERVENTION
        if any(word in query_lower for word in ['cure', 'solution', 'what should i do', 
                                               'how to fix', 'treatment', 'exercise',
                                               'technique', 'practice', 'help me']):
            return CounselingPhase.INTERVENTION
        
        # Check if we've asked enough assessment questions
        if self.assessment_count >= self.max_assessment_questions:
            return CounselingPhase.INTERVENTION
        
        # Check conversation history for previously described issues
        if len(self.conversation_context) >= 2:
            last_responses = self.conversation_context[-2:]
            issue_keywords = ['feel', 'symptom', 'problem', 'issue', 'anxious', 
                            'depressed', 'stress', 'panic', 'fear', 'worry']
            for resp in last_responses:
                if any(keyword in resp.lower() for keyword in issue_keywords):
                    return CounselingPhase.INTERVENTION
        
        return CounselingPhase.ASSESSMENT
    
    def update_session_tracking(self, query, response):
        """Track counseling session progress"""
        query_lower = query.lower()
        
        # Detect if client described an issue
        issue_keywords = ['feel', 'symptom', 'problem', 'issue', 'anxious', 
                         'depressed', 'stress', 'panic', 'fear', 'worry',
                         'cant', 'cannot', 'struggle', 'difficulty']
        
        if any(keyword in query_lower for keyword in issue_keywords):
            if not self.client_issue_type:
                # Extract main issue type
                if any(word in query_lower for word in ['anxious', 'panic', 'fear', 'worry', 'anxiety', 'nervous']):
                    self.client_issue_type = 'anxiety'
                elif any(word in query_lower for word in ['depressed', 'sad', 'hopeless', 'empty']):
                    self.client_issue_type = 'depression'
                elif any(word in query_lower for word in ['stress', 'overwhelmed', 'pressure']):
                    self.client_issue_type = 'stress'
        
        # Count assessment questions in our responses
        question_words = ['can you tell', 'what is', 'how do you', 'when did',
                         'could you', 'would you', 'tell me about', 'can you describe']
        
        if any(question_word in response.lower() for question_word in question_words):
            self.assessment_count += 1
    
    def extract_issue_summary(self):
        """Extract issue summary from conversation"""
        if not self.conversation_context:
            return "Not enough information yet"
        
        # Look for issue descriptions in last 3-4 messages
        recent_context = self.conversation_context[-4:] if len(self.conversation_context) > 4 else self.conversation_context
        
        issue_parts = []
        for msg in recent_context:
            if 'Client' in msg and any(word in msg.lower() for word in ['feel', 'symptom', 'problem', 'anxious', 'depressed']):
                clean_msg = msg.replace('Client: ', '').replace('(neutral): ', '')
                issue_parts.append(clean_msg)
        
        return " | ".join(issue_parts[:3]) if issue_parts else "Exploring concerns"
    
    def get_phase_specific_instructions(self, phase, query):
        """Get instructions specific to current phase"""
        if phase == CounselingPhase.ASSESSMENT:
            remaining = self.max_assessment_questions - self.assessment_count
            if remaining <= 0:
                return "MOVE TO INTERVENTION NOW. Provide therapeutic techniques/exercises."
            return f"ASK ONLY {remaining} MORE QUESTION(S) BEFORE MOVING TO INTERVENTION. Focus on key information only."
        
        elif phase == CounselingPhase.INTERVENTION:
            return "PROVIDE THERAPEUTIC INTERVENTION NOW. Include:\n1. Specific technique/exercise\n2. Step-by-step instructions\n3. Expected benefits\n4. When to use it"
        
        elif phase == CounselingPhase.EXPLORATION:
            return "Explore deeper thoughts/feelings. Ask reflective questions."
        
        return ""
    
    # ========== ENHANCED STYLE DETECTION ==========
    
    def determine_counseling_style(self, emotion, query):
        """Enhanced style detection for counseling role-play"""
        query_lower = query.lower()
        
        # Check for counseling session requests
        counseling_keywords = ['session', 'counselor', 'counseling', 'therapy session', 
                             'mental state', 'ask me questions', 'proper session',
                             'therapist', 'psychologist', 'talk therapy']
        
        if any(keyword in query_lower for keyword in counseling_keywords):
            self.stats["counseling_sessions_started"] += 1
            return CounselingStyle.COUNSELOR, ResponseTone.COUNSELING
        
        # Check for therapy/exercise requests
        therapy_keywords = ['exercise', 'technique', 'practice', 
                           'breathing', 'grounding', 'meditation', 'coping skill', 'cure']
        
        if any(keyword in query_lower for keyword in therapy_keywords):
            return CounselingStyle.THERAPEUTIC, ResponseTone.PRACTICAL
        
        # Check for crisis
        crisis_keywords = ['suicide', 'kill myself', 'end my life', 'self-harm',
                          'want to die', 'cant go on', 'emergency', 'urgent']
        
        if any(keyword in query_lower for keyword in crisis_keywords):
            return CounselingStyle.CRISIS, ResponseTone.CRISIS
        
        # Emotion-based styles
        if emotion in [EmotionState.DEPRESSED.value, EmotionState.ANXIOUS.value,
                      EmotionState.ANGRY.value, EmotionState.STRESSED.value,
                      EmotionState.AGITATED.value]:
            return CounselingStyle.EMPATHIC, ResponseTone.EMPATHETIC
        
        elif emotion in [EmotionState.CALM.value, EmotionState.HAPPY.value]:
            return CounselingStyle.PERSONALITY, ResponseTone.INFORMATIVE
        
        else:
            return CounselingStyle.GENERAL, ResponseTone.SUPPORTIVE
    
    # ========== RATE LIMITING METHODS ==========
    
    def enforce_rate_limit(self, estimated_tokens=100):
        """Enforce rate limits before API call"""
        now = datetime.now()
        current_date = now.date()
        
        # Reset daily counters if date changed
        if current_date != self.rate_limit_state["last_reset_date"]:
            self.rate_limit_state["daily_request_count"] = 0
            self.rate_limit_state["daily_token_count"] = 0
            self.rate_limit_state["last_reset_date"] = current_date
        
        # Check daily request limit
        if self.rate_limit_state["daily_request_count"] >= self.rate_limit_config["requests_per_day"]:
            wait_time = 60
            print(f"   ⚠️ Daily limit reached. Waiting {wait_time}s...")
            time.sleep(wait_time)
            return self.enforce_rate_limit(estimated_tokens)
        
        # Clean old timestamps
        one_minute_ago = now - timedelta(seconds=60)
        self.rate_limit_state["request_timestamps"] = [
            ts for ts in self.rate_limit_state["request_timestamps"] 
            if ts > one_minute_ago
        ]
        
        # Check RPM limit
        if len(self.rate_limit_state["request_timestamps"]) >= self.rate_limit_config["requests_per_minute"]:
            oldest_request = self.rate_limit_state["request_timestamps"][0]
            seconds_to_wait = 60 - (now - oldest_request).total_seconds()
            if seconds_to_wait > 0:
                print(f"   ⚠️ RPM limit reached. Waiting {seconds_to_wait:.1f}s...")
                time.sleep(seconds_to_wait + 0.5)
        
        # Enforce minimum interval
        if self.rate_limit_state["request_timestamps"]:
            last_request = self.rate_limit_state["request_timestamps"][-1]
            time_since_last = (now - last_request).total_seconds()
            if time_since_last < self.rate_limit_config["min_request_interval"]:
                wait_time = self.rate_limit_config["min_request_interval"] - time_since_last
                print(f"   ⏳ Enforcing interval: {wait_time:.1f}s")
                time.sleep(wait_time)
        
        # Update state
        self.rate_limit_state["request_timestamps"].append(datetime.now())
        self.rate_limit_state["daily_request_count"] += 1
        self.rate_limit_state["daily_token_count"] += estimated_tokens
        
        return True
    
    def handle_rate_limit_error(self, error):
        """Handle 429 rate limit errors"""
        self.stats["rate_limit_hits"] += 1
        
        if "retryDelay" in str(error):
            retry_match = re.search(r'retryDelay":\s*"(\d+)', str(error))
            if retry_match:
                retry_seconds = int(retry_match.group(1))
                print(f"   🔄 Rate limit hit. Retrying in {retry_seconds}s")
                time.sleep(retry_seconds + 2)
                return True
        
        backoff_seconds = min(2 ** self.stats["rate_limit_hits"], 60)
        print(f"   🔄 Rate limit hit. Backoff: {backoff_seconds}s")
        time.sleep(backoff_seconds)
        return True
    
    # ========== SEARCH & CONTEXT METHODS ==========
    
    def create_search_filters(self, counseling_style, query):
        """Create Pinecone filters with valid operators"""
        if counseling_style == CounselingStyle.THERAPEUTIC:
            return {
                "$or": [
                    {"category": {"$in": ["therapy", "exercises", "techniques"]}},
                    {"type": {"$in": ["exercise", "technique"]}}
                ]
            }
        elif counseling_style == CounselingStyle.EMPATHIC:
            return {
                "$or": [
                    {"category": {"$in": ["anxiety", "depression", "stress", "coping"]}},
                    {"type": {"$eq": "practical"}}
                ]
            }
        elif counseling_style == CounselingStyle.PERSONALITY:
            return {
                "category": {"$in": ["personality", "theory", "research"]}
            }
        
        return None
    
    def search_relevant_content(self, query, counseling_style, top_k=5):
        """Search Pinecone for relevant content"""
        try:
            query_embedding = self.embed_model.encode([query])[0].tolist()
            filters = self.create_search_filters(counseling_style, query)
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filters,
                namespace=""
            )
            
            self.stats["queries_made"] += 1
            
            if results.matches:
                return results.matches
            else:
                # Fallback without filters
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                return results.matches if results.matches else []
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def extract_key_insights(self, search_results, max_insights=3):
        """Extract key insights from search results"""
        insights = []
        
        for match in search_results[:max_insights]:
            text = match.metadata.get("text", "")
            source = match.metadata.get("source", "Unknown")
            score = match.score
            
            if score < 0.3:
                continue
            
            sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
            
            if sentences:
                selected_sentence = sentences[0]
                for sentence in sentences:
                    if len(sentence) > 50 and len(sentence) < 200:
                        selected_sentence = sentence
                        break
                
                insights.append({
                    "text": selected_sentence,
                    "source": source,
                    "relevance": round(score, 3)
                })
        
        return insights
    
    def build_context_string(self, search_results, max_results=3):
        """Build context string from search results"""
        context_parts = []
        
        for i, match in enumerate(search_results[:max_results]):
            text = match.metadata.get("text", "").strip()
            source = match.metadata.get("source", "Unknown Source")
            
            if not text:
                continue
            
            clean_text = re.sub(r'\s+', ' ', text)
            clean_text = clean_text[:500]
            
            if len(text) > 500:
                clean_text += " [...]"
            
            context_parts.append(f"[Source: {source}]\n{clean_text}")
        
        if not context_parts:
            return "General counseling knowledge."
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_conversation_context(self, limit=3):
        """Get recent conversation for context"""
        if not self.conversation_context:
            return "Beginning of conversation."
        
        recent = self.conversation_context[-limit:]
        return "\n".join([f"Previous: {msg}" for msg in recent])
    
    # ========== RESPONSE GENERATION METHODS (UPDATED) ==========
    
    def assess_risk_level(self, query, emotion):
        """Assess risk level for safety warnings"""
        query_lower = query.lower()
        
        high_risk_keywords = [
            'suicide', 'kill myself', 'end my life', 'self-harm',
            'want to die', 'harm myself', 'cant take it anymore'
        ]
        
        medium_risk_keywords = [
            'depressed', 'hopeless', 'worthless', 'crying all day',
            'cant get out of bed', 'extreme anxiety', 'panic attack'
        ]
        
        if any(keyword in query_lower for keyword in high_risk_keywords):
            return "high"
        elif (any(keyword in query_lower for keyword in medium_risk_keywords) or
              emotion in [EmotionState.DEPRESSED.value, EmotionState.ANXIOUS.value,
                         EmotionState.AGITATED.value]):
            return "medium"
        else:
            return "low"
    
    def store_interaction(self, query, emotion, style, response):
        """Store interaction for context"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "emotion": emotion,
            "style": style,
            "phase": self.counseling_phase.value,  # NEW: Store phase
            "assessment_count": self.assessment_count,  # NEW: Store count
            "response_preview": response[:100] + "..."
        }
        
        self.session_history.append(interaction)
        self.conversation_context.append(f"Client ({emotion}): {query[:50]}...")
        
        if len(self.conversation_context) > 5:
            self.conversation_context.pop(0)
    
    def generate_simple_response(self, query, emotion, search_results):
        """Fallback simple response generation"""
        insights = self.extract_key_insights(search_results)
        
        if emotion in [EmotionState.DEPRESSED.value, EmotionState.ANXIOUS.value,
                      EmotionState.AGITATED.value]:
            response = "I hear you're feeling this way, and that sounds really challenging. "
            response += "Based on psychology resources:\n\n"
        else:
            response = "Here's some information that might help:\n\n"
        
        if insights:
            for i, insight in enumerate(insights, 1):
                response += f"{i}. {insight['text']}\n"
        else:
            for match in search_results[:2]:
                text = match.metadata.get("text", "")[:200]
                if text:
                    response += f"• {text}...\n"
        
        risk_level = self.assess_risk_level(query, emotion)
        response += self.safety_warnings.get(risk_level, "")
        
        self.store_interaction(query, emotion, "fallback", response)
        
        return response
    
    def generate_counseling_response(self, query, emotion, search_results):
        """Generate counseling response with FLOW CONTROL"""
        # Determine counseling phase FIRST
        self.counseling_phase = self.determine_counseling_phase(query, emotion)
        
        # Track phase in stats
        self.stats["counseling_phases"][self.counseling_phase.value] = \
            self.stats["counseling_phases"].get(self.counseling_phase.value, 0) + 1
        
        counseling_style, response_tone = self.determine_counseling_style(emotion, query)
        
        # Update stats
        self.stats["counseling_style"][counseling_style.value] = \
            self.stats["counseling_style"].get(counseling_style.value, 0) + 1
        self.stats["response_tone"][response_tone.value] = \
            self.stats["response_tone"].get(response_tone.value, 0) + 1
        
        # Build context
        context = self.build_context_string(search_results)
        conversation_context = self.get_conversation_context()
        
        # Extract issue summary
        issue_summary = self.extract_issue_summary()
        
        # Prepare therapy exercises (always prepare for intervention phase)
        therapy_exercises_text = ""
        if self.counseling_phase == CounselingPhase.INTERVENTION or 'cure' in query.lower():
            # Use client's issue type or default to anxiety
            issue_type = self.client_issue_type or 'anxiety'
            exercises = self.therapy_exercises.get(issue_type, [])
            if exercises:
                # Format exercises nicely
                exercise_list = []
                for i, exercise in enumerate(exercises[:2], 1):  # Show max 2 exercises
                    exercise_list.append(f"{i}. {exercise['name']}:")
                    exercise_list.append(f"   Purpose: {exercise['purpose']}")
                    exercise_list.append(f"   Steps:")
                    for step in exercise['steps']:
                        exercise_list.append(f"   - {step}")
                    exercise_list.append(f"   Duration: {exercise['duration']}")
                    exercise_list.append("")
                therapy_exercises_text = "\n".join(exercise_list)
                if exercises:
                    self.counseling_tools_used.append(exercises[0]['name'])
        
        # Get phase-specific instructions
        phase_instructions = self.get_phase_specific_instructions(self.counseling_phase, query)
        
        # Get prompt template
        prompt_template = self.counseling_prompts.get(
            counseling_style.value,
            self.counseling_prompts[CounselingStyle.GENERAL.value]
        )
        
        # Fill prompt template
        prompt = prompt_template.format(
            emotion=emotion,
            query=query,
            context=context,
            conversation_context=conversation_context,
            therapy_exercises=therapy_exercises_text,
            counseling_phase=self.counseling_phase.value,
            symptoms_summary=issue_summary,
            issue_duration="Recent" if self.assessment_count < 2 else "Ongoing",
            issue_impact="Affecting social interactions" if 'people' in query.lower() else "General impact",
            phase_specific_instructions=phase_instructions
        )
        
        # Estimate tokens and enforce rate limit
        estimated_tokens = len(prompt.split()) // 0.75
        self.enforce_rate_limit(estimated_tokens)
        
        # Generate response with Gemini
        if self.use_gemini and self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                
                generated_text = response.text
                
                # Update session tracking AFTER generating response
                self.update_session_tracking(query, generated_text)
                
                # Add safety warning based on risk
                risk_level = self.assess_risk_level(query, emotion)
                safety_warning = self.safety_warnings.get(risk_level, "")
                
                final_response = generated_text + safety_warning
                self.store_interaction(query, emotion, counseling_style.value, final_response)
                
                return final_response
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    self.handle_rate_limit_error(error_msg)
                    try:
                        response = self.gemini_client.models.generate_content(
                            model=self.gemini_model,
                            contents=prompt
                        )
                        return response.text + self.safety_warnings.get(self.assess_risk_level(query, emotion), "")
                    except:
                        print(f"⚠️ Retry failed, using fallback")
                        return self.generate_fallback_counseling_response(query, emotion, search_results, self.counseling_phase)
                else:
                    print(f"⚠️ Gemini generation failed: {e}")
                    return self.generate_fallback_counseling_response(query, emotion, search_results, self.counseling_phase)
        
        # Fallback response with proper flow
        return self.generate_fallback_counseling_response(query, emotion, search_results, self.counseling_phase)
    
    def generate_fallback_counseling_response(self, query, emotion, search_results, phase):
        """Fallback response that follows counseling flow"""
        query_lower = query.lower()
        
        # If client asks for cure/solution, provide intervention immediately
        if 'cure' in query_lower or 'solution' in query_lower or 'what should i do' in query_lower:
            response = "I understand you're looking for practical solutions. Based on what you've described, here's an evidence-based technique you can try:\n\n"
            
            # Provide appropriate exercise based on described issue
            if 'people' in query_lower or 'social' in query_lower or 'judgment' in query_lower or 'suffocating' in query_lower:
                response += "🔹 **Social Anxiety Breathing Exercise:**\n"
                response += "1. When you feel suffocated around people, notice your breath\n"
                response += "2. Breathe in slowly for 4 seconds\n"
                response += "3. Hold for 2 seconds\n"
                response += "4. Breathe out slowly for 6 seconds\n"
                response += "5. Repeat 5 times\n"
                response += "This activates your calming system and reduces physical anxiety symptoms.\n\n"
                response += "🔹 **Cognitive Reframing:**\n"
                response += "When you fear judgment, ask yourself: 'What's the evidence they're judging me?'\n"
                response += "Often, we judge ourselves more harshly than others do.\n"
            else:
                # General anxiety exercise
                response += "🔹 **5-4-3-2-1 Grounding Technique:**\n"
                response += "1. Name 5 things you can SEE\n"
                response += "2. Name 4 things you can FEEL\n"
                response += "3. Name 3 things you can HEAR\n"
                response += "4. Name 2 things you can SMELL\n"
                response += "5. Name 1 thing you can TASTE\n"
                response += "This brings you back to the present moment.\n"
            
            response += "\nTry this next time you feel those symptoms. Would you like to try it now together?"
            self.counseling_phase = CounselingPhase.INTERVENTION
        
        # If in intervention phase or we've asked enough questions
        elif phase == CounselingPhase.INTERVENTION or self.assessment_count >= self.max_assessment_questions:
            response = "Thank you for sharing that. It sounds like you're dealing with anxiety in social situations. Let me provide you with some practical techniques:\n\n"
            response += "1. **Progressive Exposure:** Start by being around 1-2 people briefly, then gradually increase\n"
            response += "2. **Breathing Anchor:** Place a hand on your stomach, breathe deeply, focus on the rise/fall\n"
            response += "3. **Mantra:** Repeat 'This feeling will pass' when anxious\n\n"
            response += "Which of these would you like to try first?"
            self.counseling_phase = CounselingPhase.INTERVENTION
        
        # If still in assessment phase
        else:
            # Ask only ONE focused question
            if 'people' in query_lower or 'social' in query_lower:
                response = "I understand social situations feel suffocating. To help you better, could you tell me: When you're with people, what specific thoughts go through your mind?"
            else:
                response = "To understand how to help you best, could you describe what happens right before you start feeling this way?"
            
            self.assessment_count += 1
            self.counseling_phase = CounselingPhase.ASSESSMENT
        
        # Add safety warning
        risk_level = self.assess_risk_level(query, emotion)
        response += self.safety_warnings.get(risk_level, "")
        
        self.store_interaction(query, emotion, "fallback_counseling", response)
        return response
    
    # ========== INTEGRATION & UTILITY METHODS ==========
    
    def get_response_for_integration(self, query, emotion, return_sources=False):
        """
        Main integration method for frontend/emotion system
        """
        counseling_style, response_tone = self.determine_counseling_style(emotion, query)
        search_results = self.search_relevant_content(query, counseling_style, top_k=4)
        
        if not search_results:
            return {
                "response": "I'd like to understand what you're experiencing. Could you tell me more about how you're feeling?",
                "counseling_style": counseling_style.value,
                "response_tone": response_tone.value,
                "counseling_phase": self.counseling_phase.value,
                "assessment_count": self.assessment_count,
                "sources": [],
                "risk_level": "low",
                "timestamp": datetime.now().isoformat()
            }
        
        response = self.generate_counseling_response(query, emotion, search_results)
        
        # Extract sources
        sources = []
        if return_sources:
            for match in search_results[:3]:
                sources.append({
                    "source": match.metadata.get("source", "Unknown"),
                    "category": match.metadata.get("category", "general"),
                    "relevance": round(match.score, 3)
                })
        
        risk_level = self.assess_risk_level(query, emotion)
        
        return {
            "response": response,
            "counseling_style": counseling_style.value,
            "response_tone": response_tone.value,
            "counseling_phase": self.counseling_phase.value,
            "assessment_count": self.assessment_count,
            "client_issue_type": self.client_issue_type,
            "sources": sources,
            "query_emotion": emotion,
            "risk_level": risk_level,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_rate_limit_status(self):
        """Get current rate limit status"""
        now = datetime.now()
        requests_last_min = len([ts for ts in self.rate_limit_state["request_timestamps"] 
                                if (now - ts).total_seconds() < 60])
        
        return {
            "requests_last_minute": requests_last_min,
            "daily_requests_used": self.rate_limit_state["daily_request_count"],
            "daily_requests_left": max(0, self.rate_limit_config["requests_per_day"] - 
                                      self.rate_limit_state["daily_request_count"])
        }
    
    def show_stats(self):
        """Display system statistics"""
        print("\n" + "=" * 70)
        print("📊 COUNSELING SYSTEM STATISTICS")
        print("=" * 70)
        
        print(f"📅 Session started: {self.stats['session_start']}")
        print(f"💭 Total queries: {self.stats['queries_made']}")
        print(f"🧑‍⚕️ Counseling sessions started: {self.stats['counseling_sessions_started']}")
        print(f"🚫 Rate limit hits: {self.stats['rate_limit_hits']}")
        
        print(f"\n🔄 Counseling Flow Status:")
        print(f"   • Current phase: {self.counseling_phase.value}")
        print(f"   • Assessment questions asked: {self.assessment_count}/{self.max_assessment_questions}")
        print(f"   • Client issue type: {self.client_issue_type or 'Not identified'}")
        print(f"   • Tools used: {', '.join(self.counseling_tools_used[:3]) if self.counseling_tools_used else 'None'}")
        
        if self.stats['counseling_phases']:
            print("\n🎭 Counseling Phases Used:")
            for phase, count in self.stats['counseling_phases'].items():
                print(f"  • {phase}: {count}")
        
        rate_status = self.get_rate_limit_status()
        print(f"\n🎯 Rate Limit Status:")
        print(f"   • Requests (last min): {rate_status['requests_last_minute']}/{self.rate_limit_config['requests_per_minute']}")
        print(f"   • Requests (today): {rate_status['daily_requests_used']}/{self.rate_limit_config['requests_per_day']}")
        
        if self.stats['counseling_style']:
            print("\n🎭 Counseling Styles Used:")
            for style, count in self.stats['counseling_style'].items():
                print(f"  • {style}: {count}")
        
        print("=" * 70)
    
    def reset_session(self):
        """Reset counseling session tracking"""
        self.counseling_phase = CounselingPhase.ASSESSMENT
        self.assessment_count = 0
        self.client_issue_type = None
        self.counseling_tools_used = []
        self.conversation_context = []
        print("🔄 Counseling session reset")
    
    def chat_interface(self, emotion_input=None):
        """Interactive chat interface"""
        print("\n" + "=" * 70)
        print("💬 COUNSELING SESSION CHAT (WITH FLOW CONTROL)")
        print("=" * 70)
        print("Commands: 'quit', 'stats', 'emotion <state>', 'help', 'reset'")
        print("Emotions: depressed, anxious, angry, stressed, calm, happy, neutral, agitated")
        print("\n💡 Try: 'I'd like a counseling session' or 'Can you ask me some questions?'")
        print("💡 System will automatically move from questions to therapy after 2-3 questions")
        print("=" * 70)
        
        current_emotion = emotion_input or EmotionState.NEUTRAL.value
        
        while True:
            try:
                user_input = input("\nClient: ").strip()
                
                if user_input.lower() == 'quit':
                    print("\n👋 Thank you for our conversation. Take care.")
                    break
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                elif user_input.lower() == 'reset':
                    self.reset_session()
                    print("✅ Session reset. Starting fresh...")
                    continue
                elif user_input.lower() == 'help':
                    print("\nCounseling Session Guide:")
                    print("  quit - End session")
                    print("  stats - Show session statistics")
                    print("  reset - Reset the counseling session")
                    print("  emotion <state> - Set your current emotion")
                    print("  help - Show this guide")
                    print("\nYou can ask for:")
                    print("  • A counseling session")
                    print("  • Therapy exercises (system will provide immediately)")
                    print("  • Help with specific emotions")
                    print("  • Information about mental health")
                    continue
                elif user_input.lower().startswith('emotion '):
                    emotion = user_input[8:].strip()
                    if emotion in [e.value for e in EmotionState]:
                        current_emotion = emotion
                        print(f"✅ Emotion noted: {current_emotion}")
                    else:
                        print(f"❌ Invalid emotion")
                    continue
                
                if not user_input:
                    continue
                
                print(f"   🔍 [Noted emotion: {current_emotion}]")
                
                counseling_style, _ = self.determine_counseling_style(current_emotion, user_input)
                search_results = self.search_relevant_content(user_input, counseling_style)
                
                if not search_results:
                    print("\n🤖 Counselor: I'd like to understand more about what you're experiencing. Could you tell me more?")
                    continue
                
                response = self.generate_counseling_response(user_input, current_emotion, search_results)
                
                print(f"\n🤖 Counselor [Style: {counseling_style.value}, Phase: {self.counseling_phase.value}]:")
                print("-" * 60)
                print(response)
                print("-" * 60)
                
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n👋 Session ended. Remember to practice self-care.")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)[:100]}")

# ==========================================================
# 3. MAIN EXECUTION
# ==========================================================

def main():
    """Main function with counselor focus"""
    print("\n" + "=" * 70)
    print("🚀 ENHANCED PSYCHOLOGY RAG - COUNSELOR MODE (WITH FLOW CONTROL)")
    print("=" * 70)
    
    try:
        print("\n🔄 Initializing Counseling System...")
        
        rag = EnhancedPsychologyRAG(
            index_name="psychology-fyp",
            use_gemini=True,
            gemini_model="gemini-2.5-flash-lite"
        )
        
        print("\n✅ COUNSELING SYSTEM READY!")
        print(f"📚 Psychology Database: Connected")
        print(f"🤖 Counselor AI: ACTIVE")
        print(f"🔄 Flow Control: ACTIVE (max {rag.max_assessment_questions} questions before intervention)")
        print(f"⏱️  Rate Limiting: ACTIVE")
        
        # Test the system
        print("\n🧪 Testing Counseling Flow...")
        test_result = rag.get_response_for_integration(
            query="I'd like to start a counseling session for my social anxiety",
            emotion="anxious",
            return_sources=True
        )
        
        print(f"✅ Test successful!")
        print(f"   Style: {test_result['counseling_style']}")
        print(f"   Phase: {test_result['counseling_phase']}")
        print(f"   Response preview: {test_result['response'][:100]}...")
        
        # Ask for demo

        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

# ==========================================================
# 4. RUN THE SYSTEM
# ==========================================================

if __name__ == "__main__":
    main()





from flask import Flask, request, jsonify
import os
import time
from datetime import datetime, timedelta
import re

# --------------------------
# 1. Paste your EnhancedPsychologyRAG class here
# --------------------------
# All of your pipeline code from Kaggle should go here!
# You can literally copy all cells and paste inside this cell
# Ensure environment variables (like HF_TOKEN, Pinecone keys) are accessed via os.environ
# Example:
# HF_TOKEN = os.environ.get("HF_TOKEN")
# Then anywhere in your class, use HF_TOKEN

# For demonstration, I'm creating a minimal placeholder
class EnhancedPsychologyRAG:
    def __init__(self, index_name, use_gemini=True, gemini_model="gemini-2.5-flash-lite"):
        self.index_name = index_name
        self.use_gemini = use_gemini
        self.gemini_model = gemini_model
        self.counseling_phase = "ASSESSMENT"
        self.assessment_count = 0

    def get_response_for_integration(self, query, emotion="neutral", return_sources=False):
        # Minimal mock behavior; replace with full pipeline logic
        response = f"Received query: '{query}' with emotion '{emotion}'"
        return {
            "response": response,
            "counseling_style": "EMPATHIC",
            "response_tone": "SUPPORTIVE",
            "counseling_phase": self.counseling_phase,
            "assessment_count": self.assessment_count,
            "client_issue_type": None,
            "sources": [] if not return_sources else [{"source": "MockSource"}],
            "query_emotion": emotion,
            "risk_level": "low",
            "timestamp": datetime.now().isoformat()
        }

# --------------------------
# 2. Flask API
# --------------------------

app = Flask(__name__)

# Initialize your pipeline once (important!)
rag = EnhancedPsychologyRAG(
    index_name="psychology-fyp",
    use_gemini=True,
    gemini_model="gemini-2.5-flash-lite"
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json

    query = data.get("query")
    emotion = data.get("emotion", "neutral")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Call your RAG pipeline
    result = rag.get_response_for_integration(
        query=query,
        emotion=emotion,
        return_sources=True
    )

    return jsonify(result)

if __name__ == "__main__":
    # Use 0.0.0.0 for Render / Railway to expose the port
    app.run(host="0.0.0.0", port=10000)
