// src/components/RecognizedWords.tsx

import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface RecognizedWordsProps {
  words: string[];
  isActive: boolean;
<<<<<<< HEAD
  onForceTranslate?: () => void; // 🔹 강제 번역 함수
}

export function RecognizedWords({ words, isActive, onForceTranslate }: RecognizedWordsProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const displayWords = words.slice(-10);

=======
}

export function RecognizedWords({ words, isActive }: RecognizedWordsProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // 최근 10개 단어만 표시
  const displayWords = words.slice(-10);

  // 최신 단어가 추가될 때마다 오른쪽으로 스크롤
>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollLeft = containerRef.current.scrollWidth;
    }
  }, [displayWords]);

  return (
    <div
      className="rounded-lg shadow-md p-4 border border-gray-300"
      style={{ backgroundColor: '#9bbbd4' }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-gray-900 font-semibold">인식된 단어</h3>
<<<<<<< HEAD

        <div className="flex items-center gap-2">
          {isActive && (
            <>
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-700 font-medium">감지중</span>
            </>
          )}

          {/* 🔹 강제로 번역 버튼 */}
          {onForceTranslate && (
            <button
              onClick={onForceTranslate}
              className="ml-2 px-3 py-1 bg-blue-600 text-white text-sm rounded-full hover:bg-blue-700 transition"
            >
              번역 시작
            </button>
          )}
        </div>
=======
        {isActive && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-700 font-medium">감지중</span>
          </div>
        )}
>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330
      </div>

      <div
        ref={containerRef}
        className="flex gap-2 overflow-x-auto whitespace-nowrap p-2"
      >
        <AnimatePresence mode="popLayout">
          {displayWords.length > 0 ? (
            displayWords.map((word, index) => (
              <motion.div
                key={`${word}-${index}`}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0, opacity: 0 }}
                transition={{ type: 'spring', stiffness: 500, damping: 25 }}
                className="inline-flex px-4 py-2 bg-white text-blue-800 rounded-full border border-blue-200 shadow-sm"
              >
                {word}
              </motion.div>
            ))
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-gray-700 italic text-center w-full"
            >
              {isActive
                ? '수어를 기다리는 중...'
                : '번역 시작을 누르면 단어가 인식됩니다.'}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
