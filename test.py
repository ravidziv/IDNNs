from idnns import information_network as inet
#args = inet.get_default_parser()
net = inet.informationNetwork()
net.print_information()


#net.print_information()
print ('Starting running the network')
net.run_network()
#net.inferance()
#print ('Starting calculating information')
print ('Saving data')
net.save_data()
print ('Finished calculations')
