import React, { useState, useRef, useEffect, useCallback } from 'react';
import { X, Sparkles, Mic, MicOff, Send, BadgeCheck, Loader2, Facebook, Instagram, Youtube, Twitter } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';


// ─── Native MediaRecorder hook ────────────────────────────────────────────
function useMicRecorder({ onTranscript, onError }) {
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      const mr = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        setRecording(false);

        const blob = new Blob(chunksRef.current, { type: mimeType });
        if (blob.size < 1000) {
          onError('Recording too short. Hold the mic and speak clearly.');
          return;
        }

        setTranscribing(true);
        try {
          const formData = new FormData();
          formData.append('audio', blob, 'recording.webm');

          const res = await fetch('http://localhost:8000/api/transcribe', {
            method: 'POST',
            body: formData,
          });

          if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Transcription failed.' }));
            onError(err.detail || 'Transcription failed.');
            return;
          }

          const data = await res.json();
          if (data.transcript) {
            onTranscript(data.transcript);
          } else {
            onError('I could not hear you. Please try again.');
          }
        } catch (e) {
          onError('Could not reach the server. Is the backend running?');
        } finally {
          setTranscribing(false);
        }
      };

      // No timeslice - safest for WebM header generation
      mr.start();
      mediaRecorderRef.current = mr;
      setRecording(true);
    } catch (e) {
      if (e.name === 'NotAllowedError') {
        onError('Microphone access denied. Please allow mic in browser settings.');
      } else if (e.name === 'NotFoundError') {
        onError('No microphone found. Please connect a mic and try again.');
      } else {
        onError(`Mic error: ${e.message}`);
      }
    }
  }, [onTranscript, onError]);

  const stop = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      // 500ms delay to prevent cutting off the last word
      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
        }
      }, 500);
    }
  }, []);

  return { recording, transcribing, start, stop };
}

// ─── Main App ────────────────────────────────────────────────────────────────
function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am your Galaxy AI assistant. How can I help you today?',
      isIntro: true,
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [micError, setMicError] = useState('');
  const [micStatus, setMicStatus] = useState('idle'); // idle | recording | transcribing
  const [isChatOpen, setIsChatOpen] = useState(false);

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);


  // ── Send message ──────────────────────────────────────────────────────────
  const handleSend = useCallback(async (textOverride) => {
    const text = (textOverride ?? query).trim();
    if (!text || isLoading) return;

    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setQuery('');
    setMicError('');
    setIsLoading(true);

    try {
      const history = messages
        .filter(m => !m.isIntro)
        .map(m => ({ role: m.role, content: m.content }));

      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, history }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = '';

      // Initialize the empty bot message
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (!dataStr.trim()) continue;

            try {
              const data = JSON.parse(dataStr);

              if (data.type === 'chunk') {
                assistantMessage += data.text;
                setMessages(prev => {
                  const arr = [...prev];
                  arr[arr.length - 1] = { ...arr[arr.length - 1], content: assistantMessage };
                  return arr;
                });
              } else if (data.type === 'metadata') {
                setMessages(prev => {
                  const arr = [...prev];
                  arr[arr.length - 1] = { ...arr[arr.length - 1], content: assistantMessage, metadata: data };
                  return arr;
                });

              } else if (data.type === 'error') {
                assistantMessage = data.text;
                setMessages(prev => {
                  const arr = [...prev];
                  arr[arr.length - 1] = { ...arr[arr.length - 1], content: assistantMessage };
                  return arr;
                });
              }
            } catch (e) {
              console.error("Error parsing stream chunk", e);
            }
          }
        }
      }
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I encountered an error connecting to Galaxy AI. Please ensure the backend server is running.' },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [query, messages, isLoading, ttsEnabled]);

  // ── Mic recorder ────────────────────────────────────────────────────────
  const { recording, transcribing, start: startRecording, stop: stopRecording } = useMicRecorder({
    onTranscript: (text) => {
      setMicStatus('idle');
      setQuery(text);
      inputRef.current?.focus();
      // Auto-send after short delay so user can see the text
      setTimeout(() => {
        handleSendRef.current(text);
      }, 400);
    },
    onError: (msg) => {
      setMicStatus('idle');
      setMicError(msg);
    },
  });

  // Keep a stable ref to handleSend to avoid stale closures in async callbacks
  const handleSendRef = useRef(handleSend);
  useEffect(() => { handleSendRef.current = handleSend; }, [handleSend]);

  useEffect(() => {
    if (recording) setMicStatus('recording');
    else if (transcribing) setMicStatus('transcribing');
    else setMicStatus('idle');
  }, [recording, transcribing]);

  const toggleMic = () => {
    setMicError('');
    if (recording) {
      stopRecording();
    } else {
      startRecording();
    }
  };


  const micLabel = micStatus === 'recording' ? 'Tap to stop' : micStatus === 'transcribing' ? 'Transcribing…' : 'Speak';

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="landing-page">
      {/* ── Top Alert Bar ── */}
      <div className="top-alert-bar">
        <span>Free shipping on all orders over Rs. 5,000 | </span>
        <a href="#offers" className="alert-link">Shop latest offers</a>
      </div>

      {/* ── Mock Samsung Navbar ── */}
      <nav className="samsung-nav">
        <div className="samsung-nav-logo">SAMSUNG</div>
        <div className="samsung-nav-links">
          <span className="nav-link">Shop</span>
          <span className="nav-link">Mobile</span>
          <span className="nav-link">TV & Audio</span>
          <span className="nav-link">Appliances</span>
          <span className="nav-link">Computing</span>
          <span className="nav-link">Displays</span>
          <span className="nav-link">Support</span>
        </div>
      </nav>

      {/* ── Scrollable Content ── */}
      <main className="landing-content">
        {/* Hero Section */}
        <div className="hero-section">
          <div className="hero-background"></div>
          <div className="hero-content">
            <h1 className="hero-title">Galaxy S25 Ultra</h1>
            <p className="hero-subtitle">
              The ultimate Galaxy experience. Now with next-generation capabilities powered by Galaxy AI.
            </p>
            <div className="hero-buttons">
              <button className="hero-btn primary" onClick={() => { setIsChatOpen(true); setQuery("Tell me about the Galaxy S25 Ultra and its AI features"); }}>Learn more</button>
              <button className="hero-btn secondary">Buy now</button>
            </div>
          </div>
        </div>

        {/* Product Grid Section */}
        <section className="product-section">
          <h2 className="section-title">Recommended for you</h2>
          <div className="product-grid">
            {/* Card 1 */}
            <div className="product-card">
              <div className="product-image-container">
                <img src="/images/s25_hero.png" alt="Galaxy S25 Ultra" />
              </div>
              <div className="product-info">
                <h3>Galaxy S25 Ultra</h3>
                <p>The new era of Galaxy AI</p>
                <button className="buy-btn">Buy now</button>
              </div>
            </div>
            {/* Card 2 */}
            <div className="product-card">
              <div className="product-image-container">
                <img src="/images/watch_ultra.png" alt="Galaxy Watch Ultra" />
              </div>
              <div className="product-info">
                <h3>Galaxy Watch Ultra</h3>
                <p>Peak performance wearable</p>
                <button className="buy-btn">Buy now</button>
              </div>
            </div>
            {/* Card 3 */}
            <div className="product-card">
              <div className="product-image-container">
                <img src="/images/neo_qled.png" alt="Neo QLED 8K" />
              </div>
              <div className="product-info">
                <h3>75" Neo QLED 8K</h3>
                <p>AI-powered excellence</p>
                <button className="buy-btn">Buy now</button>
              </div>
            </div>
            {/* Card 4 */}
            <div className="product-card">
              <div className="product-image-container">
                <img src="/images/buds3_pro.png" alt="Galaxy Buds3 Pro" />
              </div>
              <div className="product-info">
                <h3>Galaxy Buds3 Pro</h3>
                <p>Immersive AI audio studio</p>
                <button className="buy-btn">Buy now</button>
              </div>
            </div>
          </div>
        </section>

        {/* Secondary Promo Banner */}
        <section className="secondary-promo">
          <div className="promo-text">
            <h2>Experience Galaxy AI</h2>
            <p>Unlock new possibilities with intelligent tools designed for your life.</p>
            <button className="promo-link" onClick={() => { setIsChatOpen(true); setQuery("What can Galaxy AI do?"); }}>Explore AI Features</button>
          </div>
        </section>

        {/* Footer */}
        <footer className="samsung-footer">
          <div className="footer-top">
            <div className="footer-brand-section">
              <div className="samsung-nav-logo">SAMSUNG</div>
              <div className="footer-tagline">Innovation and you. Powered by Galaxy AI.</div>
              <div className="footer-socials">
                <div className="social-icon"><Facebook size={20} /></div>
                <div className="social-icon"><Instagram size={20} /></div>
                <div className="social-icon"><Youtube size={20} /></div>
                <div className="social-icon"><Twitter size={20} /></div>
              </div>
            </div>

            <div className="footer-columns">
              <div className="footer-col">
                <h4>Products</h4>
                <span>Smartphones</span>
                <span>Tablets</span>
                <span>Audio Sound</span>
                <span>Watches</span>
                <span>Smart Switch</span>
              </div>
              <div className="footer-col">
                <h4>Shop</h4>
                <span>Offers</span>
                <span>Samsung Experience Store</span>
                <span>Education Store</span>
                <span>Business Store</span>
              </div>
              <div className="footer-col">
                <h4>Support</h4>
                <span>Live Chat</span>
                <span>Email Support</span>
                <span>Phone Support</span>
                <span>Community</span>
              </div>
              <div className="footer-col">
                <h4>Account</h4>
                <span>My Page</span>
                <span>My Orders</span>
                <span>My Products</span>
                <span>Sign In/Sign Up</span>
              </div>
            </div>
          </div>

          <div className="footer-bottom">
            <div className="footer-legal">
              <span>Privacy Policy</span>
              <span>Terms & Conditions</span>
              <span>Legal</span>
              <span>Sitemap</span>
            </div>
            <p>Copyright © 1995-2026 SAMSUNG All Rights reserved.</p>
          </div>
        </footer>
      </main>

      {/* ── Floating Action Button ── */}
      {!isChatOpen && (
        <button className="chat-fab" onClick={() => setIsChatOpen(true)}>
          <div className="fab-icon-container">
            <Sparkles size={22} color="#fff" />
          </div>
          Galaxy Support
        </button>
      )}

      {/* ── Chat Widget Overlay ── */}
      {isChatOpen && (
        <div className="widget-overlay">
          <div className="app-container">
            {/* Header */}
            <header className="header">
              <span className="header-brand">
                <Sparkles size={16} /> Galaxy AI
              </span>
              <div className="header-controls">
                <button className="header-icon" onClick={() => setIsChatOpen(false)}>
                  <X size={20} />
                </button>
              </div>
            </header>

            {/* Chat */}
            <div className="chat-area">
              {messages.map((msg, idx) => (
                <div key={idx}>
                  {msg.role === 'user' ? (
                    <div className="msg-user-container">
                      <div className="msg-user">{msg.content}</div>
                    </div>
                  ) : (
                    <div className="msg-bot-container">
                      <div className="bot-header">
                        <div className="sparkle-icon">
                          <Sparkles size={14} />
                        </div>
                        <span>GALAXY AI</span>
                      </div>
                      <div className="bot-card">
                        {msg.isIntro && <div className="bot-card-title">Welcome to Galaxy Support</div>}
                        <div className="bot-card-text">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              table: ({ node, ...props }) => (
                                <div className="table-wrapper">
                                  <table {...props} />
                                </div>
                              )
                            }}
                          >
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                        {msg.content?.toLowerCase().includes('s26') && (
                          <img src="/camera.png" alt="Galaxy S26" className="bot-card-image" />
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="msg-bot-container">
                  <div className="bot-header">
                    <div className="sparkle-icon"><Sparkles size={14} /></div>
                    <span>GALAXY AI IS THINKING…</span>
                  </div>
                  <div className="bot-card loading-card">
                    <Loader2 className="animate-spin" size={24} color="#1c52ec" />
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Quick chips - Dynamic for web search fallback */}
            <div className="scroll-actions">
              {messages.length > 0 &&
                messages[messages.length - 1].role === 'assistant' &&
                messages[messages.length - 1].content?.toLowerCase().includes('web search') ? (
                <>
                  <div className="action-pill primary-pill" onClick={() => { setQuery('Yes'); handleSend('Yes'); }}>Yes, search the web</div>
                  <div className="action-pill" onClick={() => setQuery("No, it's okay")}>No, thanks</div>
                </>
              ) : (
                <>
                  <div className="action-pill" onClick={() => setQuery('What are Galaxy S24 Ultra key specs?')}>S24 Ultra Specs</div>
                  <div className="action-pill" onClick={() => setQuery('Compare Galaxy S24 with S23 Ultra')}>Compare S24 vs S23</div>
                  <div className="action-pill" onClick={() => setQuery('How to use Circle to Search?')}>Circle to Search</div>
                  <div className="action-pill" onClick={() => setQuery('Show Galaxy AI camera features')}>AI Camera</div>
                  <div className="action-pill" onClick={() => setQuery('Galaxy S24 battery life & charging')}>Battery & Power</div>
                  <div className="action-pill" onClick={() => setQuery('How does Live Translate work?')}>Live Translate</div>
                </>
              )}
            </div>

            {/* Input + mic overlay */}
            <div className="input-container">
              {/* Status / error bar */}
              {(micError || micStatus !== 'idle') && (
                <div className={`voice-feedback ${micError ? 'voice-error' : ''}`}>
                  {micError || (micStatus === 'recording' ? '🎙️ Recording… tap mic to stop' : '⏳ Transcribing your speech…')}
                </div>
              )}

              <form className="input-box" onSubmit={(e) => { e.preventDefault(); handleSend(); }}>
                <input
                  ref={inputRef}
                  type="text"
                  placeholder={micStatus === 'recording' ? 'Recording…' : micStatus === 'transcribing' ? 'Transcribing…' : 'Ask Galaxy AI…'}
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  disabled={isLoading || micStatus !== 'idle'}
                />

                {/* Mic button */}
                <button
                  type="button"
                  id="mic-btn"
                  className={`icon-button mic-button ${micStatus === 'recording' ? 'mic-active' : ''} ${micStatus === 'transcribing' ? 'mic-busy' : ''}`}
                  onClick={toggleMic}
                  title={micLabel}
                  disabled={isLoading || micStatus === 'transcribing'}
                >
                  {micStatus === 'transcribing' ? (
                    <Loader2 className="animate-spin" size={20} color="#1c52ec" />
                  ) : micStatus === 'recording' ? (
                    <>
                      <MicOff size={20} color="#ef4444" />
                      <span className="mic-ring" />
                    </>
                  ) : (
                    <Mic size={20} />
                  )}
                </button>

                {/* Send button */}
                <button
                  type="submit"
                  id="send-btn"
                  className="icon-button send-button"
                  disabled={!query.trim() || isLoading || micStatus !== 'idle'}
                >
                  {isLoading ? (
                    <Loader2 className="animate-spin" size={18} color="white" />
                  ) : (
                    <Send size={18} color="white" style={{ marginLeft: '-2px' }} />
                  )}
                </button>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
