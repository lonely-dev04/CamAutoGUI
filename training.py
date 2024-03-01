from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np

# Sample dataset (replace with your actual dataset)
X = np.array([
    [0.15963297396127724, 0.12998114139844746, 0.1044686293145581, 0.09142267018417916, 0.324449965960653, 0.07019571855356899, 0.039096373829017754, 0.03562501644229556, 0.11701161295368198, 0.06361333707130501, 0.03778151407179413, 0.02981977404563894, 0.11641918877053885, 0.06364836490825537, 0.0309180403386321, 0.027534012184308756, 0.11402188150978093, 0.057685866710518514, 0.027038329076928852, 0.023323261966283426],
    [0.057452909125400874, 0.053643605014610476, 0.045691318752748344, 0.03527503141189961, 0.0871056967401399, 0.02577610292828103, 0.025071259865407414, 0.01750332615949352, 0.048543644686538344, 0.02196896994752311, 0.024160202690811636, 0.01775805104665985, 0.04491113855969697, 0.012507372757220048, 0.020695623234065357, 0.01795887610404277, 0.03981171245363446, 0.006980704334401714, 0.018516764583703643, 0.015977883906441566],
    [0.07918272765106402, 0.09101777968617686, 0.06540454171559254, 0.05319979281493225, 0.17484461822027547, 0.07983440746402772, 0.01811406856696274, 0.026920891843019455, 0.04988230110542717, 0.07973928681615025, 0.021183123663199455, 0.02370247805346234, 0.05297850878412409, 0.06792261544578017, 0.020883671916618243, 0.022664799276424463, 0.052820490205327544, 0.05181672374465268, 0.016147505031446574, 0.020171773794258437],
    [0.12475643248681487, 0.12761837602679305, 0.08715932848030737, 0.07061946235284487, 0.16389519721584783, 0.156522730748501, 0.028177896928110537, 0.045103337437921075, 0.11941593120288235, 0.18465898307420575, 0.0324377849850291, 0.05797242050239971, 0.10432515791754784, 0.16878429449142718, 0.02595587802948995, 0.05014206213913713, 0.0946534666121303, 0.13041535368575688, 0.018753622815930134, 0.04899187154053952],
    [0.1552584849150277, 0.12292227871629044, 0.10513491259639506, 0.08851042922892584, 0.26584345033315043, 0.056055300030495185, 0.04027831311040006, 0.036197894550558815, 0.1168677622375913, 0.05405566628053735, 0.03745812662865844, 0.0360726011753508, 0.11691228544849074, 0.055610930657772006, 0.029915436896505547, 0.03593519137733329, 0.10950546508446943, 0.05036542421591553, 0.02764632642182444, 0.029111033982596134],
    [0.12805850965723115, 0.1524858228503187, 0.11473853407597151, 0.0827823264126604, 0.2824947950607313, 0.14111402184794633, 0.08758490176326436, 0.06998141925522146, 0.31677891418438225, 0.14589493994037592, 0.09147412973952279, 0.07226790855647562, 0.3431213489551506, 0.13461537950298394, 0.08106957250290217, 0.06682959247317405, 0.32768304366876055, 0.10621598318698959, 0.06546485425618263, 0.056903663435099686],
    [0.1942423000234772, 0.2783761273980733, 0.2098560990784739, 0.1378972793322163, 0.13993092983393268, 0.11413457730922466, 0.17599676955455915, 0.07423576449785732, 0.20368152321823646, 0.11801590818726594, 0.22630980955297414, 0.05367203441991719, 0.21067495980965886, 0.08909578008740163, 0.2123869556723937, 0.047432852255192066, 0.22122057369110512, 0.059880564707691174, 0.1460689894450183, 0.0301012526189597],
    [0.06819967640489957, 0.07394094432728918, 0.055314288419594494, 0.05262219825381289, 0.14862497396286484, 0.06807189179691642, 0.0537810945323336, 0.0535776741499825, 0.15105405839152658, 0.12237836331519876, 0.07364696076305466, 0.05983852974924827, 0.26664476265965686, 0.12324550284616673, 0.0771808806070803, 0.06140410192696609, 0.29570953661935495, 0.09733474111233138, 0.06564855710930874, 0.05588288050134863]
])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # Labels for each sample (gesture IDs)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train K-Nearest Neighbors (KNN) classifier
k = 3  # Number of neighbors
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate model performance
print(classification_report(y_test, y_pred))

new_data = np.array([
     [[0.045309526942979114, 0.08119983288896851, 0.06045276868419344, 0.05094732313568884, 0.08898995770818413, 0.054570251643225196, 0.035642924102954075, 0.030608510180209093, 0.12706526651445826, 0.05782444475171324, 0.04328331793785108, 0.03403464212341386, 0.150905638149398, 0.05521560417226725, 0.03655033210100418, 0.031367891143218425, 0.15047572850070848, 0.042549566978532224, 0.02682770728860159, 0.024835198640351146, 0.04829164427220981, 0.07795696648083997, 0.057413149459212415, 0.05011118337331436, 0.09400270578368487, 0.05371238472304168, 0.028383758081668882, 0.017211211986102867, 0.10644421139621248, 0.05991183060429844, 0.03636173971693922, 0.023555690086608183, 0.13797554967136288, 0.0669965618087825, 0.046859379153490595, 0.0389050105722943, 0.18083970498432184, 0.05679229183847982, 0.040596374295655845, 0.03679086240345919], [0.045309526942979114, 0.08119983288896851, 0.06045276868419344, 0.05094732313568884, 0.08898995770818413, 0.054570251643225196, 0.035642924102954075, 0.030608510180209093, 0.12706526651445826, 0.05782444475171324, 0.04328331793785108, 0.03403464212341386, 0.150905638149398, 0.05521560417226725, 0.03655033210100418, 0.031367891143218425, 0.15047572850070848, 0.042549566978532224, 0.02682770728860159, 0.024835198640351146, 0.04829164427220981, 0.07795696648083997, 0.057413149459212415, 0.05011118337331436, 0.09400270578368487, 0.05371238472304168, 0.028383758081668882, 0.017211211986102867, 0.10644421139621248, 0.05991183060429844, 0.03636173971693922, 0.023555690086608183, 0.13797554967136288, 0.0669965618087825, 0.046859379153490595, 0.0389050105722943, 0.18083970498432184, 0.05679229183847982, 0.040596374295655845, 0.03679086240345919]]
])

# Predict the classes of the new data using the trained KNN classifier
predicted_classes = model.predict(new_data)

# Print the predicted classes
print("Predicted classes:", predicted_classes)
