<<<<<<< HEAD
import React, { useState, useEffect, useRef } from 'react';
=======
import React, { useState } from 'react';
>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330
import { CameraFeed } from './components/CameraFeed';
import { RecognizedWords } from './components/RecognizedWords';
import { TranslationDisplay } from './components/TranslationDisplay';
import { ControlPanel } from './components/ControlPanel';
<<<<<<< HEAD
import { motion, AnimatePresence } from 'framer-motion';
=======
>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330

export default function App() {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [recognizedWords, setRecognizedWords] = useState<string[]>([]);
<<<<<<< HEAD
  const [translations, setTranslations] = useState<string[]>([]);
  const [currentTranslation, setCurrentTranslation] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const lastUpdateTime = useRef<number>(Date.now());
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // ---------------------------
  // 카메라 토글
  // ---------------------------
=======
  const [translations, setTranslations] = useState<string[]>([]); // 빈 상태로 유지

>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330
  const handleCameraToggle = () => {
    setIsCameraActive((prev) => !prev);
    if (isCameraActive) setIsTranslating(false);
  };

<<<<<<< HEAD
  // ---------------------------
  // 번역 토글
  // ---------------------------
=======
>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330
  const handleTranslationToggle = () => {
    if (!isCameraActive) setIsCameraActive(true);
    setIsTranslating((prev) => !prev);
  };

<<<<<<< HEAD
  // ---------------------------
  // 기록 초기화
  // ---------------------------
  const handleClearHistory = () => {
    setRecognizedWords([]);
    setTranslations([]);
    setCurrentTranslation('');
  };

  // ---------------------------
  // 단어 인식 처리
  // ---------------------------
=======
  const handleClearHistory = () => {
    setRecognizedWords([]);
    setTranslations([]);
  };

>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330
  const handleFrame = (data: { detected_sign: string }) => {
    const word = data.detected_sign;
    if (!word) return;

    setRecognizedWords((prev) => {
<<<<<<< HEAD
      if (prev[prev.length - 1] === word) return prev;
      return [...prev, word].slice(-10);
    });

    lastUpdateTime.current = Date.now();
  };

  // ---------------------------
  // LLM 요청 공통 함수
  // ---------------------------
  const requestTranslation = async () => {
    if (recognizedWords.length === 0) return;

    setIsProcessing(true);
    try {
      const response = await fetch("http://localhost:8000/generate_translation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recognized_words: recognizedWords }),
      });
      const data = await response.json();
      const result = data.translated_sentence || '번역 실패';
      setCurrentTranslation(result);
      setTranslations((prev) => [...prev, result]);
    } catch (err) {
      console.error('번역 요청 실패:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  // ---------------------------
  // 자동 3초 번역
  // ---------------------------
  useEffect(() => {
    if (!isTranslating || recognizedWords.length === 0) return;

    if (timeoutRef.current) clearTimeout(timeoutRef.current);

    timeoutRef.current = setTimeout(() => {
      const elapsed = Date.now() - lastUpdateTime.current;
      if (elapsed >= 3000) {
        requestTranslation();
      }
    }, 3000);

    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [recognizedWords, isTranslating]);

=======
      return [...prev, word].slice(-10); // 최근 10개만 유지
    });

    // 현재는 번역 기능 비활성화 상태
    // setTranslations(prev => [...prev, translatedWord].slice(-10));
  };

>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330
  return (
    <div className="min-h-screen bg-gray-200 p-6 font-sans">
      <header className="text-center mb-6">
        <h1 className="text-3xl font-bold">손TALK💬</h1>
        <p className="text-lg">실시간 수어 인식</p>
      </header>

      <div className="flex gap-6">
        <div className="flex-1">
          <CameraFeed
            isActive={isCameraActive}
            isTranslating={isTranslating}
            onToggle={handleCameraToggle}
            onFrame={handleFrame}
          />

<<<<<<< HEAD
          <div className="mt-4 space-y-4 relative">
            <RecognizedWords
              words={recognizedWords}
              isActive={isTranslating}
              onForceTranslate={requestTranslation} // 🔹 버튼 클릭 시 즉시 번역
            />

            {/* 🔹 LLM 처리 중 시각화 */}
            {/* <AnimatePresence>
              {isProcessing && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="text-center text-blue-600 font-semibold"
                >
                  🧠 LLM이 문장을 분석 중입니다...
                </motion.div>
              )}
            </AnimatePresence> */}

            <TranslationDisplay
              translations={translations}
              currentTranslation={currentTranslation}
              isTranslating={isTranslating}
=======
          <div className="mt-4 space-y-4">
            <RecognizedWords words={recognizedWords} isActive={isTranslating} />

            {/* TranslationDisplay는 현재 빈 상태로 추가 */}
            <TranslationDisplay
              translations={translations}
              currentTranslation=""
              isTranslating={false}
>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330
            />
          </div>
        </div>

        <div className="w-80">
          <ControlPanel
            isTranslating={isTranslating}
            onToggleTranslation={handleTranslationToggle}
            onClearHistory={handleClearHistory}
            tone="formal"
            onToneChange={() => {}}
          />
        </div>
      </div>
    </div>
  );
}
