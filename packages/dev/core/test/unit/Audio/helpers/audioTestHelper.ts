export class AudioTestHelper {
    /**
     * Advance timers by 500ms to trigger the Sound class's timeout used for double-checking the audio context state.
     *
     * 
     */
    static WaitForAudioContextSuspendedDoubleCheck() {
        vi.advanceTimersByTime(500);
    }

    static WhenAudioContextResumes(callback: () => void) {
        return Promise.resolve().then(callback);
    }
}
