from shared_structures import RequestPacket
def generate_mock_requests():
    requests = []

    requests.append(RequestPacket(
        request_id=1,
        arrival_time=0.0,
        actual_blocks=10,
        predicted_mu=9.0,
        predicted_sigma=1.0
    ))
    requests.append(RequestPacket(
        request_id=2,
        arrival_time=1.0,
        actual_blocks=20,
        predicted_mu=18.0,
        predicted_sigma=2.0
    ))
    requests.append(RequestPacket(
        request_id=3,
        arrival_time=3.0,
        actual_blocks=15,
        predicted_mu=17,
        predicted_sigma=0.8
    ))
    requests.append(RequestPacket(
        request_id=4,
        arrival_time=3.5,
        actual_blocks=25,
        predicted_mu=22,
        predicted_sigma=0.8
    ))
    requests.append(RequestPacket(
        request_id=5,
        arrival_time=4.0,
        actual_blocks=15,
        predicted_mu=16,
        predicted_sigma=1.5
    ))

    return requests