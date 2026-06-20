import { useMemo, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import {
  Camera,
  Check,
  ChevronDown,
  ImagePlus,
  MapPin,
  Search,
  ShieldAlert,
  Sparkles,
  X,
} from "lucide-react-native";
import { StatusBar } from "expo-status-bar";

const API_URL = "https://donations-decision-specials-integer.trycloudflare.com";

const COUNTRIES = [
  "Sin region",
  "Algeria",
  "Angola",
  "Argentina",
  "Australia",
  "Bangladesh",
  "Belize",
  "Benin",
  "Bhutan",
  "Bolivia",
  "Botswana",
  "Brazil",
  "Burma",
  "Burundi",
  "Cambodia",
  "Canada",
  "China",
  "Colombia",
  "Costa Rica",
  "Democratic Republic of the Congo",
  "Ecuador",
  "El Salvador",
  "Ethiopia",
  "Fiji",
  "France",
  "French Guiana",
  "Gabon",
  "Germany",
  "Guam",
  "Guatemala",
  "Guyana",
  "Honduras",
  "Hong Kong",
  "Hong Kong S.A.R.",
  "Hungary",
  "India",
  "Indonesia",
  "Italy",
  "Ivory Coast",
  "Japan",
  "Kenya",
  "Laos",
  "Macau S.A.R",
  "Malaysia",
  "Mali",
  "Mexico",
  "Morocco",
  "Mozambique",
  "Namibia",
  "Netherlands",
  "Nicaragua",
  "Nigeria",
  "Oman",
  "Palau",
  "Panama",
  "Papua New Guinea",
  "Peru",
  "Philippines",
  "Portugal",
  "Rwanda",
  "Saudi Arabia",
  "Singapore",
  "South Africa",
  "South Sudan",
  "Spain",
  "Sri Lanka",
  "Sudan",
  "Suriname",
  "Swaziland",
  "Sweden",
  "Switzerland",
  "Taiwan",
  "Tanzania",
  "Thailand",
  "Tonga",
  "Trinidad and Tobago",
  "Tunisia",
  "Uganda",
  "United Kingdom",
  "United States of America",
  "Vanuatu",
  "Venezuela",
  "Vietnam",
  "Zambia",
  "Zimbabwe",
];

const QUICK_COUNTRIES = ["Colombia", "Mexico", "Brazil", "United States of America"];

const toneStyles = {
  red: {
    background: "#fff1f0",
    border: "#ff6b5f",
    chip: "#ffdfda",
    text: "#a72820",
  },
  green: {
    background: "#ecfdf3",
    border: "#39b86a",
    chip: "#d9fbe7",
    text: "#176c3a",
  },
  amber: {
    background: "#fff7df",
    border: "#f0aa28",
    chip: "#ffedbd",
    text: "#8b5a00",
  },
};

export default function App() {
  const [image, setImage] = useState(null);
  const [country, setCountry] = useState("");
  const [countryModalOpen, setCountryModalOpen] = useState(false);
  const [countrySearch, setCountrySearch] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const selectedTone = useMemo(() => {
    return toneStyles[result?.decision?.tone] || toneStyles.amber;
  }, [result]);

  const filteredCountries = useMemo(() => {
    const query = countrySearch.trim().toLowerCase();
    if (!query) {
      return COUNTRIES;
    }
    return COUNTRIES.filter((item) => item.toLowerCase().includes(query));
  }, [countrySearch]);

  async function pickImage(source) {
    setError("");
    setResult(null);

    const permission =
      source === "camera"
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      setError("Permiso denegado para acceder a la imagen.");
      return;
    }

    const picker =
      source === "camera"
        ? await ImagePicker.launchCameraAsync({
            allowsEditing: true,
            aspect: [1, 1],
            quality: 0.9,
          })
        : await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [1, 1],
            quality: 0.9,
          });

    if (!picker.canceled) {
      setImage(picker.assets[0]);
    }
  }

  function selectCountry(value) {
    setCountry(value === "Sin region" ? "" : value);
    setCountryModalOpen(false);
    setCountrySearch("");
  }

  async function analyzeImage() {
    if (!image) {
      setError("Selecciona o toma una foto primero.");
      return;
    }

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("country", country.trim());
    formData.append("image", {
      uri: image.uri,
      name: "snake.jpg",
      type: "image/jpeg",
    });

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
        headers: {
          "bypass-tunnel-reminder": "true",
        },
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "No se pudo analizar la imagen.");
      }
      setResult(payload);
    } catch (apiError) {
      setError(
        `No hubo conexion con el backend. Revisa que la API este en ${API_URL}. Detalle: ${apiError.message}`,
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={styles.keyboard}
      >
        <ScrollView contentContainerStyle={styles.container}>
          <View style={styles.hero}>
            <View style={styles.heroCopy}>
              <View style={styles.kickerRow}>
                <Sparkles color="#f28c35" size={16} />
                <Text style={styles.kicker}>Analisis visual</Text>
              </View>
              <Text style={styles.title}>Identificador de serpientes</Text>
              <Text style={styles.subtitle}>
                Toma una foto, elige region y revisa el riesgo con apoyo del modelo.
              </Text>
            </View>
            <View style={styles.heroBadge}>
              <ShieldAlert color="#163522" size={24} />
            </View>
          </View>

          <View style={styles.previewPanel}>
            {image ? (
              <>
                <Image source={{ uri: image.uri }} style={styles.previewImage} />
                <Pressable
                  accessibilityLabel="Quitar imagen"
                  onPress={() => {
                    setImage(null);
                    setResult(null);
                  }}
                  style={styles.clearImageButton}
                >
                  <X color="#17301f" size={18} />
                </Pressable>
              </>
            ) : (
              <View style={styles.emptyPreview}>
                <View style={styles.emptyIcon}>
                  <ImagePlus color="#246943" size={36} />
                </View>
                <Text style={styles.emptyTitle}>Agrega una imagen</Text>
                <Text style={styles.emptyText}>Camara o galeria</Text>
              </View>
            )}
          </View>

          <View style={styles.actionRow}>
            <IconButton
              icon={<Camera color="#17301f" size={22} />}
              label="Camara"
              onPress={() => pickImage("camera")}
            />
            <IconButton
              icon={<ImagePlus color="#17301f" size={22} />}
              label="Galeria"
              onPress={() => pickImage("gallery")}
            />
          </View>

          <View style={styles.fieldBlock}>
            <View style={styles.labelRow}>
              <MapPin color="#496150" size={17} />
              <Text style={styles.label}>Pais o region</Text>
            </View>
            <Pressable
              accessibilityRole="button"
              onPress={() => setCountryModalOpen(true)}
              style={({ pressed }) => [
                styles.countrySelect,
                pressed && styles.buttonPressed,
              ]}
            >
              <Text style={[styles.countryValue, !country && styles.placeholder]}>
                {country || "Seleccionar region"}
              </Text>
              <ChevronDown color="#496150" size={20} />
            </Pressable>
            <ScrollView
              horizontal
              contentContainerStyle={styles.quickList}
              showsHorizontalScrollIndicator={false}
            >
              {QUICK_COUNTRIES.map((item) => (
                <Pressable
                  key={item}
                  onPress={() => setCountry(item)}
                  style={[
                    styles.quickChip,
                    country === item && styles.quickChipSelected,
                  ]}
                >
                  <Text
                    style={[
                      styles.quickChipText,
                      country === item && styles.quickChipTextSelected,
                    ]}
                  >
                    {displayCountry(item)}
                  </Text>
                </Pressable>
              ))}
            </ScrollView>
          </View>

          <Pressable
            accessibilityRole="button"
            disabled={loading}
            onPress={analyzeImage}
            style={({ pressed }) => [
              styles.primaryButton,
              pressed && styles.buttonPressed,
              loading && styles.buttonDisabled,
            ]}
          >
            {loading ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <>
                <Search color="#ffffff" size={21} />
                <Text style={styles.primaryButtonText}>Analizar imagen</Text>
              </>
            )}
          </Pressable>

          {error ? <Text style={styles.errorText}>{error}</Text> : null}

          {result ? (
            <View
              style={[
                styles.resultPanel,
                {
                  backgroundColor: selectedTone.background,
                  borderColor: selectedTone.border,
                },
              ]}
            >
              <View
                style={[
                  styles.resultChip,
                  { backgroundColor: selectedTone.chip },
                ]}
              >
                <Text style={[styles.resultStatus, { color: selectedTone.text }]}>
                  {result.decision.display.decision}
                </Text>
              </View>
              <Text style={styles.speciesName}>
                {result.species.selected_display_name}
              </Text>
              <Text style={styles.scientificName}>
                {result.species.selected_scientific_name}
              </Text>
              <View style={styles.resultStats}>
                <Metric
                  label="Confianza"
                  value={`${Math.round(result.species.selected_confidence * 100)}%`}
                />
                <Metric label="Region" value={country || "Sin region"} />
                <Metric
                  label="Modelo"
                  value={
                    result.model?.decision_policy === "B4 safety-first"
                      ? "B4 seguro"
                      : "B4"
                  }
                />
              </View>
              <Text style={styles.reason}>{result.decision.reason}</Text>
              <Text style={styles.safetyNote}>{result.safety_note}</Text>
            </View>
          ) : null}

          {result?.species?.top_candidates?.length ? (
            <View style={styles.candidates}>
              <Text style={styles.sectionTitle}>Top 5 especies candidatas</Text>
              {result.species.top_candidates.slice(0, 5).map((candidate, index) => (
                <View
                  key={`${candidate.scientific_name}-${index}`}
                  style={styles.candidateRow}
                >
                  <View style={styles.rankBox}>
                    <Text style={styles.rankText}>{index + 1}</Text>
                  </View>
                  <View style={styles.candidateText}>
                    <Text style={styles.candidateName}>{candidate.species}</Text>
                    <Text style={styles.candidateMeta}>
                      {candidate.scientific_name} - {formatRisk(candidate.risk)}
                    </Text>
                  </View>
                  <Text style={styles.confidence}>
                    {Math.round(candidate.confidence * 100)}%
                  </Text>
                </View>
              ))}
            </View>
          ) : null}
        </ScrollView>
      </KeyboardAvoidingView>

      <CountryModal
        country={country}
        countrySearch={countrySearch}
        filteredCountries={filteredCountries}
        onClose={() => setCountryModalOpen(false)}
        onSearch={setCountrySearch}
        onSelect={selectCountry}
        visible={countryModalOpen}
      />
    </SafeAreaView>
  );
}

function CountryModal({
  country,
  countrySearch,
  filteredCountries,
  onClose,
  onSearch,
  onSelect,
  visible,
}) {
  return (
    <Modal animationType="slide" transparent visible={visible} onRequestClose={onClose}>
      <View style={styles.modalOverlay}>
        <Pressable style={styles.modalBackdrop} onPress={onClose} />
        <View style={styles.sheet}>
          <View style={styles.sheetHandle} />
          <View style={styles.sheetHeader}>
            <View>
              <Text style={styles.sheetTitle}>Pais o region</Text>
              <Text style={styles.sheetSubtitle}>Listado segun el dataset activo</Text>
            </View>
            <Pressable onPress={onClose} style={styles.closeButton}>
              <X color="#17301f" size={21} />
            </Pressable>
          </View>
          <View style={styles.searchBox}>
            <Search color="#637568" size={18} />
            <TextInput
              autoCapitalize="words"
              onChangeText={onSearch}
              placeholder="Buscar pais"
              placeholderTextColor="#7a877c"
              style={styles.searchInput}
              value={countrySearch}
            />
          </View>
          <FlatList
            data={filteredCountries}
            keyExtractor={(item) => item}
            keyboardShouldPersistTaps="handled"
            renderItem={({ item }) => {
              const selected = (country || "Sin region") === item;
              return (
                <Pressable
                  onPress={() => onSelect(item)}
                  style={({ pressed }) => [
                    styles.countryOption,
                    selected && styles.countryOptionSelected,
                    pressed && styles.buttonPressed,
                  ]}
                >
                  <Text
                    style={[
                      styles.countryOptionText,
                      selected && styles.countryOptionTextSelected,
                    ]}
                  >
                    {displayCountry(item)}
                  </Text>
                  {selected ? <Check color="#1f7a48" size={20} /> : null}
                </Pressable>
              );
            }}
            style={styles.countryList}
          />
        </View>
      </View>
    </Modal>
  );
}

function IconButton({ icon, label, onPress }) {
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={({ pressed }) => [
        styles.secondaryButton,
        pressed && styles.buttonPressed,
      ]}
    >
      {icon}
      <Text style={styles.secondaryButtonText}>{label}</Text>
    </Pressable>
  );
}

function Metric({ label, value }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue} numberOfLines={1}>
        {value}
      </Text>
    </View>
  );
}

function displayCountry(value) {
  if (value === "Sin region") {
    return "Sin region";
  }
  if (value === "United States of America") {
    return "Estados Unidos";
  }
  return value;
}

function formatRisk(value) {
  if (value === "Venomous") {
    return "Venenosa";
  }
  if (value === "Non Venomous") {
    return "No venenosa";
  }
  return "Desconocido";
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#f4f7f1",
  },
  keyboard: {
    flex: 1,
  },
  container: {
    gap: 15,
    padding: 18,
    paddingBottom: 34,
  },
  hero: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#dbe4d9",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: 14,
    padding: 17,
  },
  heroCopy: {
    flex: 1,
    gap: 5,
  },
  kickerRow: {
    alignItems: "center",
    flexDirection: "row",
    gap: 6,
  },
  kicker: {
    color: "#637568",
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0,
    textTransform: "uppercase",
  },
  title: {
    color: "#102317",
    fontSize: 29,
    fontWeight: "900",
    letterSpacing: 0,
    lineHeight: 34,
  },
  subtitle: {
    color: "#536359",
    fontSize: 14,
    lineHeight: 20,
  },
  heroBadge: {
    alignItems: "center",
    backgroundColor: "#e6f3ea",
    borderColor: "#c9dfd0",
    borderRadius: 8,
    borderWidth: 1,
    height: 48,
    justifyContent: "center",
    width: 48,
  },
  previewPanel: {
    aspectRatio: 1,
    backgroundColor: "#dfe9df",
    borderColor: "#cbd8cd",
    borderRadius: 8,
    borderWidth: 1,
    overflow: "hidden",
  },
  previewImage: {
    height: "100%",
    width: "100%",
  },
  clearImageButton: {
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.88)",
    borderRadius: 8,
    height: 38,
    justifyContent: "center",
    position: "absolute",
    right: 12,
    top: 12,
    width: 38,
  },
  emptyPreview: {
    alignItems: "center",
    flex: 1,
    justifyContent: "center",
    gap: 8,
  },
  emptyIcon: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#cbd8cd",
    borderRadius: 8,
    borderWidth: 1,
    height: 76,
    justifyContent: "center",
    width: 76,
  },
  emptyTitle: {
    color: "#17301f",
    fontSize: 18,
    fontWeight: "900",
  },
  emptyText: {
    color: "#637568",
    fontSize: 13,
    fontWeight: "700",
  },
  actionRow: {
    flexDirection: "row",
    gap: 11,
  },
  secondaryButton: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#cbd8cd",
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    flexDirection: "row",
    gap: 8,
    justifyContent: "center",
    minHeight: 50,
  },
  secondaryButtonText: {
    color: "#17301f",
    fontSize: 15,
    fontWeight: "800",
  },
  fieldBlock: {
    backgroundColor: "#ffffff",
    borderColor: "#dbe4d9",
    borderRadius: 8,
    borderWidth: 1,
    gap: 10,
    padding: 12,
  },
  labelRow: {
    alignItems: "center",
    flexDirection: "row",
    gap: 6,
  },
  label: {
    color: "#344238",
    fontSize: 14,
    fontWeight: "900",
  },
  countrySelect: {
    alignItems: "center",
    backgroundColor: "#f8faf6",
    borderColor: "#cbd8cd",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    minHeight: 50,
    paddingHorizontal: 13,
  },
  countryValue: {
    color: "#102317",
    flex: 1,
    fontSize: 16,
    fontWeight: "800",
  },
  placeholder: {
    color: "#7a877c",
    fontWeight: "700",
  },
  quickList: {
    gap: 8,
    paddingRight: 2,
  },
  quickChip: {
    backgroundColor: "#eef4ec",
    borderColor: "#d7e1d6",
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 11,
    paddingVertical: 8,
  },
  quickChipSelected: {
    backgroundColor: "#163522",
    borderColor: "#163522",
  },
  quickChipText: {
    color: "#496150",
    fontSize: 12,
    fontWeight: "800",
  },
  quickChipTextSelected: {
    color: "#ffffff",
  },
  primaryButton: {
    alignItems: "center",
    backgroundColor: "#1f7a48",
    borderRadius: 8,
    flexDirection: "row",
    gap: 9,
    justifyContent: "center",
    minHeight: 54,
  },
  primaryButtonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "900",
  },
  buttonPressed: {
    opacity: 0.78,
  },
  buttonDisabled: {
    opacity: 0.68,
  },
  errorText: {
    backgroundColor: "#fff1f0",
    borderColor: "#ff8a80",
    borderRadius: 8,
    borderWidth: 1,
    color: "#a72820",
    padding: 12,
  },
  resultPanel: {
    borderRadius: 8,
    borderWidth: 1,
    gap: 10,
    padding: 16,
  },
  resultChip: {
    alignSelf: "flex-start",
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  resultStatus: {
    fontSize: 13,
    fontWeight: "900",
    textTransform: "uppercase",
  },
  speciesName: {
    color: "#102317",
    fontSize: 26,
    fontWeight: "900",
    lineHeight: 31,
  },
  scientificName: {
    color: "#344238",
    fontSize: 15,
    fontStyle: "italic",
  },
  resultStats: {
    flexDirection: "row",
    gap: 10,
  },
  metric: {
    backgroundColor: "rgba(255,255,255,0.62)",
    borderColor: "rgba(16,35,23,0.08)",
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    gap: 3,
    paddingHorizontal: 10,
    paddingVertical: 9,
  },
  metricLabel: {
    color: "#637568",
    fontSize: 11,
    fontWeight: "900",
    textTransform: "uppercase",
  },
  metricValue: {
    color: "#102317",
    fontSize: 15,
    fontWeight: "900",
  },
  reason: {
    color: "#223227",
    fontSize: 14,
    lineHeight: 20,
  },
  safetyNote: {
    color: "#102317",
    fontSize: 13,
    fontWeight: "800",
    lineHeight: 19,
  },
  candidates: {
    gap: 10,
  },
  sectionTitle: {
    color: "#102317",
    fontSize: 17,
    fontWeight: "900",
  },
  candidateRow: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#dbe4d9",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: 10,
    minHeight: 70,
    paddingHorizontal: 11,
    paddingVertical: 9,
  },
  rankBox: {
    alignItems: "center",
    backgroundColor: "#edf5ed",
    borderRadius: 8,
    height: 34,
    justifyContent: "center",
    width: 34,
  },
  rankText: {
    color: "#1f7a48",
    fontWeight: "900",
  },
  candidateText: {
    flex: 1,
    gap: 3,
  },
  candidateName: {
    color: "#102317",
    fontSize: 15,
    fontWeight: "900",
  },
  candidateMeta: {
    color: "#5f6f63",
    fontSize: 12,
    lineHeight: 16,
  },
  confidence: {
    color: "#102317",
    fontSize: 14,
    fontWeight: "900",
    minWidth: 40,
    textAlign: "right",
  },
  modalOverlay: {
    flex: 1,
    justifyContent: "flex-end",
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(16,35,23,0.34)",
  },
  sheet: {
    backgroundColor: "#ffffff",
    borderTopLeftRadius: 8,
    borderTopRightRadius: 8,
    maxHeight: "78%",
    paddingBottom: 16,
    paddingHorizontal: 16,
    paddingTop: 9,
  },
  sheetHandle: {
    alignSelf: "center",
    backgroundColor: "#c9d5cc",
    borderRadius: 8,
    height: 5,
    marginBottom: 13,
    width: 42,
  },
  sheetHeader: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  sheetTitle: {
    color: "#102317",
    fontSize: 21,
    fontWeight: "900",
  },
  sheetSubtitle: {
    color: "#637568",
    fontSize: 12,
    fontWeight: "700",
    marginTop: 2,
  },
  closeButton: {
    alignItems: "center",
    backgroundColor: "#eef4ec",
    borderRadius: 8,
    height: 38,
    justifyContent: "center",
    width: 38,
  },
  searchBox: {
    alignItems: "center",
    backgroundColor: "#f8faf6",
    borderColor: "#dbe4d9",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: 8,
    minHeight: 48,
    paddingHorizontal: 12,
  },
  searchInput: {
    color: "#102317",
    flex: 1,
    fontSize: 16,
    minHeight: 48,
  },
  countryList: {
    marginTop: 10,
  },
  countryOption: {
    alignItems: "center",
    borderRadius: 8,
    flexDirection: "row",
    justifyContent: "space-between",
    minHeight: 48,
    paddingHorizontal: 12,
  },
  countryOptionSelected: {
    backgroundColor: "#edf8f0",
  },
  countryOptionText: {
    color: "#243428",
    flex: 1,
    fontSize: 15,
    fontWeight: "700",
  },
  countryOptionTextSelected: {
    color: "#176c3a",
    fontWeight: "900",
  },
});
