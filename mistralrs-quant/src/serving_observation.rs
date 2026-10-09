use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostDispatchPhase {
    Unknown,
    Prefill,
    Decode,
}

thread_local! {
    static PHASE: std::cell::Cell<HostDispatchPhase> = const { std::cell::Cell::new(HostDispatchPhase::Unknown) };
}

pub struct HostDispatchScope {
    previous: Option<HostDispatchPhase>,
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl HostDispatchScope {
    pub fn enter(phase: HostDispatchPhase) -> Self {
        Self {
            previous: enabled().then(|| PHASE.with(|current| current.replace(phase))),
            _thread_bound: std::marker::PhantomData,
        }
    }
}

impl Drop for HostDispatchScope {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            PHASE.with(|current| current.set(previous));
        }
    }
}

pub struct HostDispatchCounter {
    total: AtomicUsize,
    phases: [AtomicUsize; 3],
}

impl Default for HostDispatchCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl HostDispatchCounter {
    pub const fn new() -> Self {
        Self {
            total: AtomicUsize::new(0),
            phases: [
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
            ],
        }
    }

    pub fn record(&self) {
        if !enabled() {
            return;
        }
        self.total.fetch_add(1, Ordering::Relaxed);
        let phase = PHASE.with(|current| current.get());
        let index = match phase {
            HostDispatchPhase::Unknown => 0,
            HostDispatchPhase::Prefill => 1,
            HostDispatchPhase::Decode => 2,
        };
        self.phases[index].fetch_add(1, Ordering::Relaxed);
    }

    pub fn total(&self) -> usize {
        self.total.load(Ordering::Relaxed)
    }

    pub fn phase_counts(&self) -> [usize; 3] {
        self.phases
            .each_ref()
            .map(|counter| counter.load(Ordering::Relaxed))
    }
}

pub static FP8_NATIVE: HostDispatchCounter = HostDispatchCounter::new();
pub static FP8_DEQUANT: HostDispatchCounter = HostDispatchCounter::new();
pub static UNQUANT_GEMV: HostDispatchCounter = HostDispatchCounter::new();
pub static UNQUANT_MATMUL: HostDispatchCounter = HostDispatchCounter::new();

pub fn counters() -> [usize; 4] {
    [&FP8_NATIVE, &FP8_DEQUANT, &UNQUANT_GEMV, &UNQUANT_MATMUL].map(|counter| counter.total())
}

static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var("MISTRALRS_INFERENCE_OBSERVATIONS").map_or(true, |value| value != "0")
    })
}

pub fn phase_counters() -> [[usize; 3]; 4] {
    [&FP8_NATIVE, &FP8_DEQUANT, &UNQUANT_GEMV, &UNQUANT_MATMUL]
        .map(|counter| counter.phase_counts())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_contract_unscoped_dispatch_is_unknown() {
        let counter = HostDispatchCounter::new();
        counter.record();
        assert_eq!(counter.phase_counts(), [1, 0, 0]);
        assert_eq!(counter.total(), 1);
    }

    #[test]
    fn observation_contract_dispatch_scope_nested_and_unwound() {
        let counter = HostDispatchCounter::new();
        counter.record();
        {
            let _prefill = HostDispatchScope::enter(HostDispatchPhase::Prefill);
            counter.record();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _decode = HostDispatchScope::enter(HostDispatchPhase::Decode);
                counter.record();
                panic!("scope unwind fixture");
            }));
            assert!(result.is_err());
            counter.record();
        }
        counter.record();
        assert_eq!(counter.total(), 5);
        assert_eq!(counter.phase_counts(), [2, 2, 1]);
    }
}
