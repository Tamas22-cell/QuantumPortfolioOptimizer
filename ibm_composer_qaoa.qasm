OPENQASM 3.0;
include "stdgates.inc";

// Quantum Portfolio Optimizer — IBM Quantum Composer-ready QAOA circuit
// Assets: q[0]=BTC, q[1]=ETH, q[2]=SOL, q[3]=NVDA
// Budget: choose 2 assets
// QAOA depth p=1
// Fixed parameters: beta=0.5, gamma=0.5
// Cost model: mean-variance objective + quadratic budget penalty

qubit[4] q;
bit[4] c;

// Initial |+> state
h q[0];
h q[1];
h q[2];
h q[3];

// Cost Hamiltonian: single-qubit Z terms
rz(-0.065) q[0];
rz(-0.059) q[1];
rz(-0.0735) q[2];
rz(-0.0225) q[3];

// Cost Hamiltonian: ZZ terms, decomposed as CX-RZ-CX
// BTC-ETH
cx q[0], q[1];
rz(0.525) q[1];
cx q[0], q[1];

// BTC-SOL
cx q[0], q[2];
rz(0.530) q[2];
cx q[0], q[2];

// BTC-NVDA
cx q[0], q[3];
rz(0.510) q[3];
cx q[0], q[3];

// ETH-SOL
cx q[1], q[2];
rz(0.5275) q[2];
cx q[1], q[2];

// ETH-NVDA
cx q[1], q[3];
rz(0.509) q[3];
cx q[1], q[3];

// SOL-NVDA
cx q[2], q[3];
rz(0.511) q[3];
cx q[2], q[3];

// Mixer Hamiltonian: RX(2*beta), beta=0.5
rx(1.0) q[0];
rx(1.0) q[1];
rx(1.0) q[2];
rx(1.0) q[3];

// Measurement
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
