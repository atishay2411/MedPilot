import { useState, useRef, useEffect } from 'react';
import { 
  Bot, User, Send, Paperclip, Plus, 
  Activity, ActivitySquare, Pill, ClipboardList, 
  Search, ShieldAlert, BadgeInfo, Mic, MicOff, HeartPulse, UserCircle2
} from 'lucide-react';

export default function App() {
  // Load initial state from LocalStorage
  const loadInitialState = (key, defaultVal) => {
    try {
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : defaultVal;
    } catch {
      return defaultVal;
    }
  };

  const [messages, setMessages] = useState(() => loadInitialState('medpilot_messages', []));
  const [patientUuid, setPatientUuid] = useState(() => loadInitialState('medpilot_patientUuid', null));
  
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [llmStatus, setLlmStatus] = useState(null);
  const [patientData, setPatientData] = useState(null);
  const [isListening, setIsListening] = useState(false);
  
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const chatEndRef = useRef(null);
  const speechRecognitionRef = useRef(null);

  // Sync to LocalStorage
  useEffect(() => {
    localStorage.setItem('medpilot_messages', JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    localStorage.setItem('medpilot_patientUuid', JSON.stringify(patientUuid));
    if (patientUuid) {
      fetchPatientSummary(patientUuid);
    } else {
      setPatientData(null);
    }
  }, [patientUuid]);

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initial Data Fetch
  useEffect(() => {
    fetch('/api/llm/status')
      .then(res => res.json())
      .then(data => setLlmStatus(data?.data))
      .catch(console.error);
      
    // Initialize Speech Recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      
      recognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }
        if (finalTranscript) {
          setInput(prev => prev + ' ' + finalTranscript.trim());
        }
      };
      
      recognition.onerror = () => setIsListening(false);
      recognition.onend = () => setIsListening(false);
      speechRecognitionRef.current = recognition;
    }
  }, []);
  
  const toggleListening = () => {
    if (isListening) {
      speechRecognitionRef.current?.stop();
    } else {
      speechRecognitionRef.current?.start();
      setIsListening(true);
    }
  };

  const fetchPatientSummary = async (uuid) => {
    try {
      const res = await fetch(`/api/patients/${uuid}/summary`);
      if (res.ok) {
        const payload = await res.json();
        if (payload.ok) setPatientData(payload.data);
      }
    } catch (e) {
      console.error("Failed to load patient summary banner", e);
    }
  };

  const clearSession = () => {
    setMessages([]);
    setPatientUuid(null);
    setInput('');
  };

  const sendMessage = async (overridePrompt = null) => {
    const textToSend = overridePrompt || input;
    if (!textToSend.trim() && !selectedFile) return;

    if (isListening && speechRecognitionRef.current) {
        speechRecognitionRef.current.stop();
        setIsListening(false);
    }

    const userMsg = { role: 'user', content: textToSend };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('prompt', textToSend);
      if (patientUuid) formData.append('patient_uuid', patientUuid);
      
      const history = messages.map(m => {
        if (typeof m.content === 'string') return { role: m.role, content: m.content };
        // Include active patient name in assistant messages so the LLM always
        // knows which patient was loaded, even after error responses.
        const patientPrefix = m.content.patient_context?.display
          ? `[Patient: ${m.content.patient_context.display}] `
          : '';
        return { role: m.role, content: patientPrefix + (m.content.message || '') };
      });
      if (history.length > 0) formData.append('history', JSON.stringify(history.slice(-14)));
      if (selectedFile) formData.append('file', selectedFile);

      const res = await fetch('/api/chat', { method: 'POST', body: formData });
      const payload = await res.json();
      
      if (res.ok && payload.ok) {
        setMessages(prev => [...prev, { role: 'assistant', content: payload.data }]);
        if (payload.data?.patient_context?.uuid) {
          setPatientUuid(payload.data.patient_context.uuid);
        }
        setSelectedFile(null);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', content: { message: `Error: ${payload.error || 'Unknown error'}` } }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: { message: 'Connection failed.' } }]);
    } finally {
      setIsLoading(false);
    }
  };

  const confirmAction = async (actionId, isDestructive) => {
    const text = isDestructive ? window.prompt("Type DELETE to confirm this destructive action:") : "";
    if (isDestructive && text !== "DELETE") return;
    
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('action_id', actionId);
      if (isDestructive) formData.append('destructive_confirm_text', text);

      const res = await fetch('/api/chat/confirm', { method: 'POST', body: formData });
      const payload = await res.json();
      if (res.ok && payload.ok) {
        setMessages(prev => [...prev, { role: 'assistant', content: payload.data }]);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', content: { message: `Failed to confirm: ${payload.error}` } }]);
      }
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: { message: 'Connection failed.' } }]);
    } finally {
      setIsLoading(false);
    }
  };

  const quickActions = [
    { icon: Search, text: "Search patient", prompt: "Find patient Maria Santos" },
    { icon: Activity, text: "Patient analysis", prompt: "Summarize this patient" },
    { icon: BadgeInfo, text: "View vitals", prompt: "Show their vitals" },
    { icon: ActivitySquare, text: "View conditions", prompt: "Show their conditions" },
    { icon: Pill, text: "View medications", prompt: "Show their medications" },
    { icon: ShieldAlert, text: "View allergies", prompt: "Show their allergies" },
    { icon: ClipboardList, text: "Add clinical note", prompt: "Record a clinical note for this patient" },
  ];

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden">
      {/* SIDEBAR */}
      <div className="w-72 bg-white border-r border-slate-200 flex flex-col shadow-sm z-20">
        <div className="p-5 flex items-center gap-3 border-b border-slate-100">
          <img src="/medpilot_logo.png" alt="Logo" className="w-12 h-12 object-contain bg-white" />
          <div>
            <h1 className="font-bold text-lg leading-tight text-slate-800">MedPilot</h1>
            <p className="text-xs text-slate-500 font-medium">AI Clinical Copilot</p>
          </div>
        </div>

        <div className="p-4 flex-1 overflow-y-auto">
          <button 
            onClick={clearSession}
            className="w-full mb-6 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2.5 rounded-lg text-sm font-medium transition-all shadow-sm"
          >
            <Plus size={16} /> New Session
          </button>

          <div className="mb-6">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 px-1">Quick Actions</h3>
            <div className="space-y-1">
              {quickActions.map((action, i) => (
                <button 
                  key={i}
                  onClick={() => sendMessage(action.prompt)}
                  className="w-full flex items-center gap-3 px-3 py-2 text-sm text-slate-600 hover:text-indigo-600 hover:bg-indigo-50 rounded-md transition-colors"
                >
                  <action.icon size={16} className="text-slate-400" />
                  <span className="font-medium text-left">{action.text}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Status Footer */}
        <div className="p-4 bg-slate-50 border-t border-slate-200 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-slate-500 font-medium">System Status</span>
            {llmStatus?.enabled ? (
              <span className="flex items-center gap-1.5 text-emerald-600 font-semibold">
                <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"></span>
                Ready ({llmStatus.provider})
              </span>
            ) : (
              <span className="text-amber-600 font-medium">Offline</span>
            )}
          </div>
        </div>
      </div>

      {/* MAIN LAYOUT */}
      <div className="flex-1 flex flex-col relative h-full">
        
        {/* DYNAMIC PATIENT BANNER */}
        {patientData && (
          <div className="absolute top-0 w-full bg-white/95 backdrop-blur-sm border-b border-slate-200 shadow-sm z-10 px-6 py-3 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center">
                <UserCircle2 size={24} />
              </div>
              <div>
                <h2 className="font-bold text-slate-800 flex items-center gap-2">
                  {patientData.patient?.given_name} {patientData.patient?.family_name}
                  <span className="text-[10px] bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full border border-slate-200 uppercase">
                    MRN: {patientData.patient?.patient_uuid?.slice(0,8)}
                  </span>
                </h2>
                <div className="text-xs text-slate-500 flex gap-3 font-medium">
                  <span>DOB: {patientData.patient?.birthdate || 'N/A'}</span>
                  <span>•</span>
                  <span>Gender: {patientData.patient?.gender?.toUpperCase() || 'N/A'}</span>
                </div>
              </div>
            </div>
            
            <div className="flex gap-6 text-sm">
              <div className="flex flex-col items-end">
                <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">Active Conditions</span>
                <span className="font-semibold text-slate-700 flex items-center gap-1">
                  <ActivitySquare size={14} className="text-indigo-400" />
                  {patientData.active_conditions?.length || 0}
                </span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">Medications</span>
                <span className="font-semibold text-slate-700 flex items-center gap-1">
                  <Pill size={14} className="text-emerald-400" />
                  {patientData.active_medications?.length || 0}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* CHAT AREA */}
        <div className={`flex-1 overflow-y-auto p-6 scroll-smooth ${patientData ? 'mt-[72px]' : ''}`}>
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-4 max-w-md mx-auto text-center">
              <div className="w-20 h-20 bg-indigo-50 rounded-full flex items-center justify-center mb-2">
                <HeartPulse size={40} className="text-indigo-300" />
              </div>
              <h2 className="text-2xl font-bold text-slate-700">Clinical AI Assistance</h2>
              <p className="text-sm">Search for a patient, upload records, or utilize voice dictation to generate insights.</p>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto space-y-8 pb-36">
              {messages.map((msg, i) => {
                const isUser = msg.role === 'user';
                return (
                  <div key={i} className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
                    <div className={`w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center shadow-sm ${isUser ? 'bg-slate-200 text-slate-600' : 'bg-white border border-slate-100 text-indigo-600'}`}>
                      {isUser ? <User size={20} /> : <Bot size={20} />}
                    </div>
                    
                    <div className={`flex flex-col max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
                      <div className={`px-5 py-3.5 rounded-2xl shadow-sm text-sm/relaxed ${
                        isUser 
                          ? 'bg-slate-800 text-white rounded-tr-none' 
                          : 'bg-white border border-slate-100 text-slate-800 rounded-tl-none'
                      }`}>
                        {typeof msg.content === 'string' ? (
                          <div className="whitespace-pre-wrap">{msg.content}</div>
                        ) : (
                          <div className="w-full">
                            {msg.content.intent && !['inform', 'clarify'].includes(msg.content.intent) && (
                              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 mb-3 rounded-full bg-indigo-50 text-indigo-700 text-xs font-bold border border-indigo-100 uppercase tracking-wide">
                                {msg.content.intent.replace('_', ' ')}
                              </div>
                            )}

                            <div className="prose prose-sm max-w-none text-slate-700 font-medium whitespace-pre-wrap mb-2">
                              {msg.content.message}
                            </div>
                            
                            {msg.content.summary && (
                              <div className="mt-3 p-3 bg-slate-50 border border-slate-100 rounded-lg text-slate-600">
                                {msg.content.summary}
                              </div>
                            )}

                            {msg.content.pending_action && (
                              <div className="mt-4 border border-rose-200 bg-rose-50 rounded-xl overflow-hidden shadow-sm">
                                <div className="px-4 py-3 bg-white border-b border-rose-100 flex items-center gap-2">
                                  <ShieldAlert size={16} className="text-rose-500" />
                                  <span className="font-bold text-slate-800 text-sm">{msg.content.pending_action.action}</span>
                                </div>
                                <div className="p-4">
                                  <pre className="text-xs bg-white p-3 rounded-lg border border-slate-100 text-slate-600 overflow-x-auto mb-4">
                                    {JSON.stringify(msg.content.pending_action.payload || msg.content.pending_action.metadata, null, 2)}
                                  </pre>
                                  <div className="flex justify-end gap-2">
                                    <button 
                                      onClick={() => confirmAction(msg.content.pending_action.id, msg.content.pending_action.destructive)}
                                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-lg shadow-sm transition-colors"
                                    >
                                      {msg.content.pending_action.destructive ? 'Authorize Destructive Action' : 'Confirm & Execute'}
                                    </button>
                                  </div>
                                </div>
                              </div>
                            )}

                            {msg.content.data && (
                               <details className="mt-3 p-3 border border-slate-200 rounded-lg group cursor-pointer hover:bg-slate-50 transition-colors">
                                 <summary className="text-xs font-bold text-slate-500 uppercase flex items-center focus:outline-none">
                                    <span>View Structured Internal Payload</span>
                                 </summary>
                                 <pre className="mt-2 text-[11px] overflow-x-auto bg-slate-900 text-slate-300 p-3 rounded-md">
                                   {JSON.stringify(msg.content.data, null, 2)}
                                 </pre>
                               </details>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              
              {isLoading && (
                <div className="flex gap-4">
                  <div className="w-10 h-10 bg-indigo-50 border border-indigo-100 rounded-full flex items-center justify-center">
                    <div className="flex gap-1.5">
                      <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        {/* INPUT TRAY */}
        <div className="absolute bottom-0 w-full bg-gradient-to-t from-slate-50 via-slate-50 to-transparent pt-10 pb-6 px-6 z-20">
          <div className="max-w-4xl mx-auto">
            {selectedFile && (
              <div className="mb-2 inline-flex items-center gap-2 px-3 py-1.5 bg-indigo-50 border border-indigo-100 text-indigo-700 rounded-lg text-xs font-semibold shadow-sm">
                <Paperclip size={14} />
                {selectedFile.name}
                <button onClick={() => setSelectedFile(null)} className="ml-2 hover:text-indigo-900">✕</button>
              </div>
            )}
            
            <div className={`relative flex items-center shadow-lg rounded-2xl bg-white border ring-1 transition-all
              ${isListening ? 'border-rose-300 ring-rose-100 shadow-rose-100' : 'border-slate-200 ring-slate-100 focus-within:ring-indigo-500 focus-within:border-indigo-500'}
            `}>
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="p-3.5 text-slate-400 hover:text-indigo-600 transition-colors ml-1"
                title="Attach PDF"
              >
                <Paperclip size={20} />
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept=".pdf" 
                  onChange={(e) => setSelectedFile(e.target.files[0])}
                />
              </button>
              
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && sendMessage()}
                placeholder={isListening ? "Listening... (Speak now)" : "Message MedPilot..."}
                className="flex-1 bg-transparent border-none py-4 text-slate-800 placeholder-slate-400 focus:outline-none text-base"
              />
              
              {/* Dictation Button */}
              {speechRecognitionRef.current && (
                <button 
                  onClick={toggleListening}
                  className={`p-2.5 mr-1 rounded-xl transition-all ${
                    isListening 
                      ? 'bg-rose-100 text-rose-600 animate-pulse shadow-inner' 
                      : 'text-slate-400 hover:bg-slate-100 hover:text-slate-600'
                  }`}
                  title="Voice Dictation"
                >
                  {isListening ? <Mic size={20} /> : <MicOff size={20} />}
                </button>
              )}

              <button 
                onClick={() => sendMessage()}
                disabled={!input.trim() && !selectedFile || isLoading}
                className="m-2 p-2.5 bg-slate-900 text-white rounded-xl hover:bg-indigo-600 disabled:opacity-50 disabled:hover:bg-slate-900 transition-all shadow-sm"
              >
                <Send size={18} className="translate-x-[-1px] translate-y-[1px]" />
              </button>
            </div>
            
            <p className="text-center text-[11px] text-slate-400 mt-3 font-medium">Verify clinical information natively in OpenMRS before executing critical actions.</p>
          </div>
        </div>
        
      </div>
    </div>
  );
}
