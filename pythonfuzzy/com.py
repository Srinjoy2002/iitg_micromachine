from pymodbus.client import ModbusSerialClient

# Configure Modbus RTU connection
client = ModbusSerialClient(
    port='COM4',  # Change to your actual COM port
    baudrate=38400,  # Match servo settings
    bytesize=8,
    parity='N',
    stopbits=2,
    timeout=1
)

SLAVE_ID = 1  # Usually 1, but check your settings
SERVO_ON_REGISTER = 0x1000  # Servo ON/OFF register

if client.connect():
    print("Connected to Delta ASDA-E3 Servo")

    # **Turn OFF the Servo**
    response = client.write_register(SERVO_ON_REGISTER, 0, slave=SLAVE_ID)  # Disengage servo
    if response.isError():
        print("Error turning off servo")
    else:
        print("Servo turned OFF")

    # Close connection
    client.close()

else:
    print("Failed to connect to Delta ASDA-E3 Servo")
