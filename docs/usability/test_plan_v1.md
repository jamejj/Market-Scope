# MarketScope Usability Test Plan v1

Cel testu: sprawdzić, czy osoba spoza projektu potrafi samodzielnie zrozumieć MarketScope i wykonać podstawowe zadania bez instrukcji krok po kroku.

Ten test nie sprawdza skuteczności inwestycyjnej modelu. Sprawdza zrozumiałość produktu, nawigację, język i zaufanie do prezentowanych wniosków.

## Zasady testu

- Nie tłumacz testerowi, gdzie ma kliknąć.
- Nie poprawiaj testera w trakcie zadania, chyba że całkowicie utknie.
- Poproś testera, żeby myślał na głos.
- Notuj pierwsze kliknięcie, miejsca zawahania i słowa, których tester nie rozumie.
- Po każdym zadaniu zapytaj: „Co Twoim zdaniem MarketScope właśnie powiedział?”.
- Nie oceniaj testera. Jeśli coś jest niejasne, to problem produktu, nie użytkownika.

## Czas

15–25 minut na jedną osobę.

## Kogo testować

Minimum 3 osoby docelowo:

- jedna osoba początkująca inwestycyjnie,
- jedna osoba inwestująca okazjonalnie,
- jedna osoba bardziej techniczna lub rynkowa.

Pierwszy test może być wykonany na jednej osobie, żeby złapać największe problemy.

## Setup przed testem

- Uruchom lokalną aplikację MarketScope.
- Upewnij się, że działa aktualny `main`.
- Nie pokazuj testerowi kodu, terminala ani historii projektu.
- Jeżeli lokalne dane forward/watchlisty są prywatne, powiedz tylko: „To jest lokalny testowy stan aplikacji”.
- Nie używaj słów typu FAST, ML, proof ani Candidate przed testerem, chyba że sam o nie zapyta.

## Co uznajemy za sukces

Tester bez naszej pomocy potrafi:

- znaleźć analizę konkretnego instrumentu,
- odczytać główny wniosek bez traktowania go jako gwarancji zysku,
- znaleźć interesujący setup z rynku,
- dodać coś do obserwowanych,
- wrócić do obserwacji i opisać zmianę tezy,
- znaleźć miejsce, gdzie MarketScope wyjaśnia, jak działa i jak jest testowany.

## Co będzie sygnałem problemu P0

- Tester rozumie wynik jako polecenie kupna/sprzedaży.
- Tester nie potrafi znaleźć analizy konkretnego symbolu.
- Tester nie rozumie różnicy między sygnałem badawczym a pozycją forward proof.
- Tester nie potrafi wrócić do zapisanej obserwacji.
- Tester myśli, że watchlista lub forward proof realnie wykonuje transakcje.

## Po teście

Po każdym teście wypełnij `feedback_template.md`. Po 1–3 testach zbierz problemy w:

- P0 — blokuje pierwszego testera/publiczny pokaz,
- P1 — mocno poprawia zrozumiałość,
- P2 — polish/backlog.
