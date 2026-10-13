/*
 * HVAC damper control with a PID loop
 * ------------------------------------
 * Simulated control stage for the TIPE study "Optimisation energetique des
 * systemes CVC par un modele hybride d'IA".
 *
 * A DHT22 sensor measures the indoor temperature. A PID controller compares it
 * to the comfort setpoint (22 degC, from ASHRAE 55 / ISO 7730) and drives a
 * servo-actuated damper between fully closed and fully open. An LED indicates
 * that the system is actively conditioning the air.
 *
 * This is the physical-control half of the study. The LSTM forecast (see
 * ../../src/) supplies the setpoint schedule in the full hybrid model; here the
 * setpoint is fixed so the control loop can be tested on its own.
 *
 * Circuit (Arduino UNO):
 *   DHT22 data  -> D2   (10k pull-up to 5V)
 *   Servo PWM   -> D9
 *   Status LED  -> D7   (through 220R to GND)
 *
 * NOTE ON PROVENANCE: the original 2023 Wokwi sketch was not preserved. This
 * file was written later to match the circuit documented in the study (see
 * figures/00_wokwi_circuit.png) and the control strategy it describes. It is a
 * faithful reimplementation, not the original source.
 */

#include <DHT.h>
#include <Servo.h>

// ---------------------------------------------------------------- pins
const uint8_t PIN_DHT   = 2;
const uint8_t PIN_SERVO = 9;
const uint8_t PIN_LED   = 7;

#define DHT_TYPE DHT22

// ------------------------------------------------------------ setpoint
// 22 degC: overlap of ASHRAE Standard 55 (20-26) and ISO 7730 (22).
const float SETPOINT_C = 22.0;

// Deadband: below this error the damper holds position, so the servo does
// not hunt around the setpoint on sensor noise.
const float DEADBAND_C = 0.3;

// ------------------------------------------------------------ PID gains
// Same structure as pid.py in this repository:
//     u(t) = Kp*e(t) + Ki*integral(e) + Kd*de/dt
const float KP = 12.0;
const float KI = 0.8;
const float KD = 4.0;

// Damper travel, in degrees of servo rotation.
const float DAMPER_MIN = 0.0;    // fully closed
const float DAMPER_MAX = 180.0;  // fully open

// Control period. 1000 ms is also the minimum sampling interval of a DHT22.
const unsigned long SAMPLE_MS = 1000;

// ---------------------------------------------------------------- state
DHT dht(PIN_DHT, DHT_TYPE);
Servo damper;

float integral   = 0.0;
float prevError  = 0.0;
unsigned long lastSample = 0;

void setup() {
  Serial.begin(115200);
  dht.begin();
  damper.attach(PIN_SERVO);
  pinMode(PIN_LED, OUTPUT);

  damper.write(DAMPER_MIN);
  digitalWrite(PIN_LED, LOW);

  Serial.println(F("HVAC damper control - PID"));
  Serial.print(F("Setpoint: "));
  Serial.print(SETPOINT_C, 1);
  Serial.println(F(" degC"));
  Serial.println(F("t_ms,temp_C,error_C,u,damper_deg"));
}

/*
 * One PID step. Returns the raw control output.
 * dt is passed in seconds so the gains stay independent of the loop period.
 */
float pidStep(float error, float dt) {
  integral += error * dt;

  // Anti-windup: clamp the integral term so a long saturation period cannot
  // accumulate an output the actuator can never deliver.
  const float INTEGRAL_LIMIT = (DAMPER_MAX - DAMPER_MIN) / max(KI, 0.001f);
  integral = constrain(integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);

  float derivative = (error - prevError) / dt;
  prevError = error;

  return KP * error + KI * integral + KD * derivative;
}

void loop() {
  unsigned long now = millis();
  if (now - lastSample < SAMPLE_MS) {
    return;
  }
  float dt = (now - lastSample) / 1000.0;
  lastSample = now;

  float tempC = dht.readTemperature();
  if (isnan(tempC)) {
    // Failed read: hold the last damper position rather than slamming it shut.
    Serial.println(F("DHT read failed"));
    return;
  }

  // Positive error means the room is warmer than the setpoint, so the damper
  // should open to supply more cooling.
  float error = tempC - SETPOINT_C;

  float damperDeg;
  bool conditioning;

  if (fabs(error) < DEADBAND_C) {
    // Inside the comfort band: close the damper and let the integral decay,
    // so the loop does not fight small fluctuations.
    integral *= 0.95;
    prevError = error;
    damperDeg = DAMPER_MIN;
    conditioning = false;
  } else {
    float u = pidStep(error, dt);
    damperDeg = constrain(u, DAMPER_MIN, DAMPER_MAX);
    conditioning = damperDeg > DAMPER_MIN;
  }

  damper.write((int)damperDeg);
  digitalWrite(PIN_LED, conditioning ? HIGH : LOW);

  Serial.print(now);          Serial.print(',');
  Serial.print(tempC, 2);     Serial.print(',');
  Serial.print(error, 2);     Serial.print(',');
  Serial.print(KP * error + KI * integral, 2); Serial.print(',');
  Serial.println(damperDeg, 0);
}
