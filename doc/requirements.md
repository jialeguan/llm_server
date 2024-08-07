## Stateless computation & Enforceable guarantees

### Requirements

>  Private Cloud Compute must use the personal user data that it receives exclusively for the purpose of fulfilling the user’s request. This data must never be available to anyone other than the user, not even to Apple staff, not even during active processing. And this data **must not be retained, including via logging or for debugging, after the response is returned to the user**. In other words, we want a strong form of stateless data processing where personal data **leaves no trace in the PCC system**.

### Operation

- All code that can run on the node must be part of a trust cache that has been signed by Apple, approved for that specific PCC node, and loaded by the Secure Enclave such that it cannot be changed or amended at runtime.

### Data

- And this data must not be retained, including via logging or for debugging, after the response is returned to the user. In other words, we want a strong form of stateless data processing where personal data leaves no trace in the PCC system.

## No privileged runtime access

- only pre-specified, structured, and audited logs and metrics can leave the node

## Non-targetability

- An attacker should not be able to attempt to compromise personal data that belongs to specific, targeted Private Cloud Compute users without attempting a broad compromise of the entire PCC system.
 In other words, a limited PCC compromise must not allow the attacker to steer requests from specific users to compromised nodes; targeting users should require a wide attack that’s likely to be detected.

- Target diffusion starts with the request metadata, which leaves out any personally identifiable information about the source device or user, and includes only limited contextual data about the request that’s required to enable routing to the appropriate model. This metadata is the only part of the user’s request that is available to load balancers and other data center components running outside of the PCC trust boundary. The metadata also includes a single-use credential, based on RSA Blind Signatures, to authorize valid requests without tying them to a specific user. Additionally, PCC requests go through an OHTTP relay — operated by a third party — which hides the device’s source IP address before the request ever reaches the PCC infrastructure. This prevents an attacker from using an IP address to identify requests or associate them with an individual. It also means that an attacker would have to compromise both the third-party relay and our load balancer to steer traffic based on the source IP address.

## Verifiable transparency
